"""多步 Workflow 评测执行器。"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from eval.execution.gateway import LLMGateway
from eval.execution.utils import RESULT_ROW_SCHEMA_VERSION
from eval.io.store import RunStore
from eval.registry import ModelRegistry, estimate_cost
from eval.tasks import build_task
from eval.workflow import WorkflowSpec

__all__ = ["run_workflow_mode"]

logger = logging.getLogger(__name__)


async def run_workflow_mode(
    *,
    run_id: str,
    store: RunStore,
    dataset: list[dict[str, Any]],
    registry: ModelRegistry,
    workflow: WorkflowSpec,
    global_params: dict[str, Any],
    gateway: LLMGateway,
    default_model_id: str = "",
    step_model_map: dict[str, str] | None = None,
    workflow_concurrency: int = 1,
) -> list[dict[str, Any]]:
    """执行 workflow 模式，输出 step 级与 e2e 级结果。

    step_model_map 优先覆盖步骤自身的 model_id；
    步骤 model_id 为 ``$active`` 时回退到 default_model_id。
    """
    _step_map = step_model_map or {}
    records: list[dict[str, Any]] = []

    async def _run_one_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
        sample_records: list[dict[str, Any]] = []
        sample_id = str(sample.get("sample_id", ""))
        step_outputs: dict[str, dict[str, Any]] = {}
        e2e_success = True
        e2e_latency = 0.0
        e2e_tokens = 0
        e2e_retry = 0
        e2e_cost = 0.0
        final_output: dict[str, Any] | None = None

        for step in workflow.steps:
            # 解析最终使用的 model_id：CLI step_model_map > 步骤显式声明 > default_model_id（$active 回退）
            raw_model_id = _step_map.get(step.id) or (
                default_model_id if step.model_id == "$active" else step.model_id
            )
            if not raw_model_id:
                raise ValueError(
                    f"Cannot resolve model for step '{step.id}': "
                    f"no --models supplied and no active model in registry."
                )
            task = build_task(step.task)
            merged_params = dict(global_params)
            merged_params.update(step.params_override)

            # step 输入可来自原始样本或上一步解析结果。
            if step.input_from == "sample":
                step_input = sample
            else:
                prev = step_outputs.get(step.input_from)
                if prev is None or not prev.get("parse_success"):
                    reason = (
                        f"upstream_{step.input_from}_parse_failed"
                        if prev is not None
                        else f"upstream_{step.input_from}_missing"
                    )
                    logger.warning(
                        "workflow step '%s' skipped: %s (sample=%s)",
                        step.id, reason, sample_id,
                    )
                    skip_row = {
                        "schema_version": RESULT_ROW_SCHEMA_VERSION,
                        "run_id": run_id,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "mode": "workflow_step",
                        "workflow_name": workflow.name,
                        "workflow_version": workflow.version,
                        "sample_id": sample_id,
                        "task_name": step.task,
                        "task_version": "skipped",
                        "step_id": step.id,
                        "prompt_version": "skipped",
                        "model_id": raw_model_id,
                        "params": merged_params,
                        "success": False,
                        "error": reason,
                        "parse_applicable": task.parse_applicable,
                        "parse_success": False,
                        "latency_ms": 0.0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "retry_count": 0,
                        "cost_estimate": 0.0,
                        "from_cache": False,
                        "output_text": "",
                        "parsed": None,
                        "task_metrics": {},
                    }
                    store.append_result(skip_row)
                    sample_records.append(skip_row)
                    step_outputs[step.id] = {"parsed": None, "raw": "", "success": False, "parse_success": False}
                    final_output = None
                    e2e_success = False
                    continue
                step_input = prev["parsed"]

            try:
                if gateway.mock:
                    messages, prompt_version = [], "mock"
                else:
                    messages, prompt_version = task.build_prompt(
                        sample, context={"input": step_input, "step_outputs": step_outputs}
                    )
            except Exception as exc:
                step_row = {
                    "schema_version": RESULT_ROW_SCHEMA_VERSION,
                    "run_id": run_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "mode": "workflow_step",
                    "workflow_name": workflow.name,
                    "workflow_version": workflow.version,
                    "sample_id": sample_id,
                    "task_name": task.name,
                    "task_version": task.version,
                    "step_id": step.id,
                    "prompt_version": "error",
                    "model_id": raw_model_id,
                    "params": merged_params,
                    "success": False,
                    "error": f"build_prompt_error: {exc}",
                    "parse_applicable": task.parse_applicable,
                    "parse_success": False,
                    "latency_ms": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "retry_count": 0,
                    "cost_estimate": 0.0,
                    "from_cache": False,
                    "output_text": "",
                    "parsed": None,
                    "task_metrics": {},
                }
                store.append_result(step_row)
                sample_records.append(step_row)
                step_outputs[step.id] = {"parsed": None, "raw": "", "success": False, "parse_success": False}
                final_output = None
                e2e_success = False
                continue

            try:
                call = await gateway.call(
                    model_id=raw_model_id,
                    task_name=task.name,
                    sample=sample,
                    sample_cache_id=f"{sample_id}:{step.id}:{task.name}",
                    messages=messages,
                    params_override=merged_params,
                )
            except Exception as exc:
                step_row = {
                    "schema_version": RESULT_ROW_SCHEMA_VERSION,
                    "run_id": run_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "mode": "workflow_step",
                    "workflow_name": workflow.name,
                    "workflow_version": workflow.version,
                    "sample_id": sample_id,
                    "task_name": task.name,
                    "task_version": task.version,
                    "step_id": step.id,
                    "prompt_version": prompt_version,
                    "model_id": raw_model_id,
                    "params": merged_params,
                    "success": False,
                    "error": f"gateway_error: {exc}",
                    "parse_applicable": task.parse_applicable,
                    "parse_success": False,
                    "latency_ms": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "retry_count": 0,
                    "cost_estimate": 0.0,
                    "from_cache": False,
                    "output_text": "",
                    "parsed": None,
                    "task_metrics": {},
                }
                store.append_result(step_row)
                sample_records.append(step_row)
                step_outputs[step.id] = {"parsed": None, "raw": "", "success": False, "parse_success": False}
                final_output = None
                e2e_success = False
                continue

            parsed = None
            parse_success = False
            task_metrics: dict[str, Any] = {}
            row_success = bool(call.success)
            error_message = call.error
            if call.success:
                try:
                    parsed = task.parse(call.content)
                    parse_success = parsed is not None
                except Exception as exc:
                    row_success = False
                    parse_success = False
                    error_message = (
                        f"{error_message}; parse_error: {exc}"
                        if error_message
                        else f"parse_error: {exc}"
                    )
            else:
                parse_success = False
            try:
                task_metrics = task.metrics(sample, call.content, parsed)
            except Exception as exc:
                row_success = False
                error_message = (
                    f"{error_message}; metrics_error: {exc}"
                    if error_message
                    else f"metrics_error: {exc}"
                )
                task_metrics = {}
            spec = registry.get(raw_model_id)
            cost = 0.0 if call.from_cache else estimate_cost(call.usage, spec.price_config)

            e2e_success = e2e_success and row_success and (parse_success or not task.parse_applicable)
            e2e_latency += call.latency_ms
            e2e_tokens += call.usage["total_tokens"]
            e2e_retry += call.retry_count
            e2e_cost += cost
            final_output = parsed

            step_row = {
                "schema_version": RESULT_ROW_SCHEMA_VERSION,
                "run_id": run_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "mode": "workflow_step",
                "workflow_name": workflow.name,
                "workflow_version": workflow.version,
                "sample_id": sample_id,
                "task_name": task.name,
                "task_version": task.version,
                "step_id": step.id,
                "prompt_version": prompt_version,
                "model_id": raw_model_id,
                "params": merged_params,
                "success": row_success,
                "error": error_message,
                "parse_applicable": task.parse_applicable,
                "parse_success": parse_success,
                "latency_ms": round(call.latency_ms, 2),
                "prompt_tokens": call.usage["prompt_tokens"],
                "completion_tokens": call.usage["completion_tokens"],
                "total_tokens": call.usage["total_tokens"],
                "retry_count": call.retry_count,
                "cost_estimate": round(cost, 8),
                "from_cache": call.from_cache,
                "output_text": call.content,
                "parsed": parsed,
                "task_metrics": task_metrics,
            }
            store.append_result(step_row)
            sample_records.append(step_row)

            step_outputs[step.id] = {
                "parsed": parsed,
                "raw": call.content,
                "success": call.success,
                "parse_success": parse_success,
            }

        # 端到端汇总行用于整体效果回归与版本对比。
        e2e_row = {
            "schema_version": RESULT_ROW_SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "mode": "workflow_e2e",
            "workflow_name": workflow.name,
            "workflow_version": workflow.version,
            "sample_id": sample_id,
            "task_name": workflow.name,
            "task_version": workflow.version,
            "step_id": "__e2e__",
            "prompt_version": "workflow",
            "model_id": "+".join(
                _step_map.get(step.id) or (default_model_id if step.model_id == "$active" else step.model_id)
                for step in workflow.steps
            ),
            "params": global_params,
            "success": e2e_success,
            "error": "" if e2e_success else "workflow_step_failed",
            "parse_applicable": True,
            "parse_success": final_output is not None,
            "latency_ms": round(e2e_latency, 2),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": e2e_tokens,
            "retry_count": e2e_retry,
            "cost_estimate": round(e2e_cost, 8),
            "from_cache": False,
            "output_text": "",
            "parsed": final_output,
            "task_metrics": {},
        }
        store.append_result(e2e_row)
        sample_records.append(e2e_row)

        return sample_records

    if workflow_concurrency <= 1:
        for sample in dataset:
            records.extend(await _run_one_sample(sample))
        return records

    sem = asyncio.Semaphore(workflow_concurrency if workflow_concurrency > 0 else 10_000)

    async def _bounded_run(sample: dict[str, Any]) -> list[dict[str, Any]]:
        async with sem:
            return await _run_one_sample(sample)

    all_rows = await asyncio.gather(*[_bounded_run(sample) for sample in dataset])
    for rows in all_rows:
        records.extend(rows)
    return records
