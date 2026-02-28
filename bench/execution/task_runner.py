"""单任务多模型评测执行器。

执行流程：对每个 (model, sample, repeat) 组合：
  1. build_prompt → 2. gateway.call → 3. parse → 4. metrics → 5. 落盘

并发设计：
- 每个模型独立 Semaphore，防止单模型触发限流。
- 流式调度（限制 in-flight 协程数），用 FIRST_COMPLETED 回收，避免全量
  asyncio.gather 导致的内存峰值。
- 支持断点续跑（completed_keys 跳过已完成项）。

错误处理：每个阶段的异常被包装为对应的 EvalBenchError 子类，
通过 derive_error_fields() 归一化后写入 ResultRow。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from bench.contracts.exceptions import (
    EvalBenchError,
    GatewayCallError,
    MetricsComputeError,
    ParseOutputError,
    PromptBuildError,
)
from bench.contracts.error import derive_error_fields
from bench.contracts.result import ResultRow
from bench.execution.gateway import LLMGateway
from bench.execution.utils import RESULT_ROW_SCHEMA_VERSION, STEP_ID_TASK, resolve_concurrency
from bench.io.store import RunStore
from bench.registry import ModelRegistry, estimate_cost
from bench.tasks import build_task

__all__ = ["run_task_mode"]

logger = logging.getLogger(__name__)


async def run_task_mode(
    *,
    run_id: str,
    store: RunStore,
    dataset: list[dict[str, Any]],
    registry: ModelRegistry,
    task_name: str,
    model_ids: list[str],
    params_override: dict[str, Any],
    repeats: int,
    gateway: LLMGateway,
    concurrency_override: int | None = None,
    completed_keys: set[tuple[str, str, str, str, int]] | None = None,
) -> list[dict[str, Any]]:
    """执行"单任务 + 多模型"评测模式，每个模型使用独立 Semaphore 并发。"""
    task = build_task(task_name)
    completed = completed_keys or set()
    records: list[dict[str, Any]] = []
    concurrency_map = resolve_concurrency(model_ids, registry, concurrency_override)
    sems: dict[str, asyncio.Semaphore] = {
        mid: asyncio.Semaphore(c if c > 0 else 10_000)
        for mid, c in concurrency_map.items()
    }
    logger.info(
        "task=%s  concurrency per model: %s",
        task_name,
        {mid: (c if c > 0 else "unlimited") for mid, c in concurrency_map.items()},
    )

    async def _call_one(
        model_id: str,
        spec: Any,
        sample: dict[str, Any],
        repeat_idx: int,
    ) -> dict[str, Any]:
        sample_id = str(sample.get("sample_id", ""))
        prompt_version = "unknown"
        messages: list[dict[str, Any]] = []
        task_defaults = dict(getattr(task, "default_params", {}) or {})
        merged_params = {**task_defaults, **params_override}

        def _error_row(error: str, *, error_exc: EvalBenchError | None = None) -> dict[str, Any]:
            fields = derive_error_fields(
                success=False,
                parse_applicable=task.parse_applicable,
                parse_success=False,
                error_message=error,
                error_exc=error_exc,
            )
            row = ResultRow(
                schema_version=RESULT_ROW_SCHEMA_VERSION,
                run_id=run_id,
                mode="task",
                sample_id=sample_id,
                task_name=task.name,
                task_version=task.version,
                step_id=STEP_ID_TASK,
                prompt_version=prompt_version,
                model_id=model_id,
                params=merged_params,
                repeat_idx=repeat_idx,
                success=False,
                error=error,
                parse_applicable=task.parse_applicable,
                parse_success=False,
                latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                retry_count=0,
                cost_estimate=0.0,
                from_cache=False,
                output_text="",
                parsed=None,
                task_metrics={},
                **fields,
            )
            out = asdict(row)
            out["timestamp"] = datetime.now(UTC).isoformat()
            return out

        async with sems[model_id]:
            try:
                if gateway.mock:
                    messages, prompt_version = [], "mock"
                else:
                    messages, prompt_version = task.build_prompt(sample, context={"repeat_idx": repeat_idx})
            except Exception as exc:
                err = PromptBuildError(str(exc))
                return _error_row(str(err), error_exc=err)
            try:
                call = await gateway.call(
                    model_id=model_id,
                    task_name=task.name,
                    sample=sample,
                    sample_cache_id=f"{sample_id}:{repeat_idx}:{task.name}",
                    messages=messages,
                    params_override=merged_params,
                )
            except Exception as exc:
                err = exc if isinstance(exc, EvalBenchError) else GatewayCallError(str(exc))
                return _error_row(str(err), error_exc=err)

        parsed = None
        parse_success = False
        task_metrics: dict[str, Any] = {}
        row_success = bool(call.success)
        error_message = call.error
        derived_exc: EvalBenchError | None = None

        if call.success:
            try:
                parsed = task.parse(call.content)
                parse_success = parsed is not None
            except Exception as exc:
                row_success = False
                parse_success = False
                parse_exc = ParseOutputError(str(exc))
                derived_exc = parse_exc
                error_message = f"{error_message}; {parse_exc}" if error_message else str(parse_exc)
        else:
            parse_success = False

        try:
            task_metrics = task.metrics(sample, call.content, parsed)
        except Exception as exc:
            row_success = False
            metrics_exc = MetricsComputeError(str(exc))
            derived_exc = metrics_exc
            error_message = f"{error_message}; {metrics_exc}" if error_message else str(metrics_exc)
            task_metrics = {}

        cost = 0.0 if call.from_cache else estimate_cost(call.usage, spec.price_config)
        fields = derive_error_fields(
            success=row_success,
            parse_applicable=task.parse_applicable,
            parse_success=parse_success,
            error_message=error_message or "",
            error_type=call.error_type,
            error_stage=call.error_stage,
            error_code=call.error_code,
            error_exc=derived_exc,
        )
        return {
            "schema_version": RESULT_ROW_SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "mode": "task",
            "sample_id": sample_id,
            "task_name": task.name,
            "task_version": task.version,
            "step_id": STEP_ID_TASK,
            "prompt_version": prompt_version,
            "model_id": model_id,
            "params": merged_params,
            "repeat_idx": repeat_idx,
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
            **fields,
        }

    # 流式调度：限制 in-flight 协程数量，避免一次性创建全量协程导致内存峰值升高。
    max_pending = max(1, sum(c if c > 0 else 128 for c in concurrency_map.values()))
    pending: set[asyncio.Task[dict[str, Any]]] = set()

    async def _drain_one() -> None:
        nonlocal pending
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task_fut in done:
            try:
                row = task_fut.result()
            except Exception as exc:
                logger.error("Unexpected task_runner coroutine exception: %s", exc, exc_info=True)
                continue
            store.append_result(row)
            records.append(row)

    for model_id in model_ids:
        spec = registry.get(model_id)
        for sample in dataset:
            for repeat_idx in range(repeats):
                sample_id = str(sample.get("sample_id", ""))
                key = (sample_id, task.name, STEP_ID_TASK, model_id, repeat_idx)
                if key in completed:
                    continue
                pending.add(asyncio.create_task(_call_one(model_id, spec, sample, repeat_idx)))
                if len(pending) >= max_pending:
                    await _drain_one()

    while pending:
        await _drain_one()

    return records
