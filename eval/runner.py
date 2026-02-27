"""评测执行入口：支持单任务评测与多步 workflow 评测。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from eval.cache import EvalCache
from eval.gateway import LLMGateway
from eval.metrics import aggregate_records
from eval.registry import ModelRegistry, estimate_cost
from eval.reporter import MarkdownReporter, Reporter
from eval.store import RunStore
from eval.tasks import build_task
from eval.workflow import WorkflowSpec, load_workflow

RESULT_ROW_SCHEMA_VERSION = "result_row.v1"
SUMMARY_ROW_SCHEMA_VERSION = "summary_row.v1"
SCORER_VERSION = "v1"


def _load_dataset(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    """加载 JSONL 数据集，可选限制样本数量。"""
    if max_samples == 0:
        return []
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def _git_commit_hash() -> str:
    """读取当前 git commit，失败时返回 unknown。"""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out
    except Exception:
        return "unknown"


def _parse_params(params_str: str | None) -> dict[str, Any]:
    """解析 CLI 传入的 JSON 参数覆盖。"""
    if not params_str:
        return {}
    return dict(json.loads(params_str))


def _parse_workflow_models(
    models_str: str,
    registry: "ModelRegistry",
) -> tuple[str, dict[str, str]]:
    """解析 workflow 模式下的 --models 参数，返回 (default_model_id, step_model_map)。

    支持两种格式：
    - 单一模型 ID：``kimi-personal``  →  所有 $active 步骤均使用该模型
    - 步骤映射：``step1_dedup=kimi-personal,step2_extract=deepseek-v3``  →  按步骤 ID 单独指定
      未出现在映射中的 $active 步骤回退到 registry.active
    """
    if not models_str:
        default = registry.active or ""
        return default, {}
    if "=" in models_str:
        step_map: dict[str, str] = {}
        for part in models_str.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                step_map[k.strip()] = v.strip()
        return registry.active or "", step_map
    return models_str.strip(), {}


def _resolve_concurrency(
    model_ids: list[str],
    registry: ModelRegistry,
    user_value: int | None,
) -> dict[str, int]:
    """返回每个 model_id 的独立并发数。

    规则（依优先级）：
    - CLI --concurrency 不为 None 时：所有模型统一设为该值
    - 否则各模型独立读取自身配置中的 ``concurrency`` 字段，缺省 8
    0 = 无限制
    """
    result: dict[str, int] = {}
    for mid in model_ids:
        if user_value is not None:
            result[mid] = user_value
        else:
            cfg_val = int(registry.get(mid).default_params.get("concurrency", 8))
            result[mid] = cfg_val
    return result


async def _run_task_mode(
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
) -> list[dict[str, Any]]:
    """执行"单任务 + 多模型"评测模式，每个模型使用独立 Semaphore 并发。"""
    task = build_task(task_name)
    records: list[dict[str, Any]] = []
    # 每个模型独立限流：0 = 无限制，用极大值模拟
    concurrency_map = _resolve_concurrency(model_ids, registry, concurrency_override)
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
        async with sems[model_id]:
            if gateway.mock:
                messages, prompt_version = [], "mock"
            else:
                messages, prompt_version = task.build_prompt(sample, context={"repeat_idx": repeat_idx})
            call = await gateway.call(
                model_id=model_id,
                task_name=task.name,
                sample=sample,
                sample_cache_id=f"{sample_id}:{repeat_idx}:{task.name}",
                messages=messages,
                params_override=params_override,
            )
        parsed = task.parse(call.content) if call.success else None
        task_metrics = task.metrics(sample, call.content, parsed)
        # P0 Fix: 缓存命中不产生 API 调用，成本应为 0
        cost = 0.0 if call.from_cache else estimate_cost(call.usage, spec.price_config)
        return {
            "schema_version": RESULT_ROW_SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "mode": "task",
            "sample_id": sample_id,
            "task_name": task.name,
            "task_version": task.version,
            "step_id": "__task__",
            "prompt_version": prompt_version,
            "model_id": model_id,
            "params": params_override,
            "repeat_idx": repeat_idx,
            "success": call.success,
            "error": call.error,
            "parse_applicable": task.parse_applicable,
            "parse_success": parsed is not None,
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

    coros = [
        _call_one(model_id, registry.get(model_id), sample, repeat_idx)
        for model_id in model_ids
        for sample in dataset
        for repeat_idx in range(repeats)
    ]
    rows = await asyncio.gather(*coros)
    for row in rows:
        store.append_result(row)
        records.append(row)

    return records


async def _run_workflow_mode(
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
                    # 若后续步骤被跳过，e2e parsed 不应沿用上一步的历史值。
                    final_output = None
                    e2e_success = False
                    continue
                step_input = prev["parsed"]

            if gateway.mock:
                messages, prompt_version = [], "mock"
            else:
                messages, prompt_version = task.build_prompt(sample, context={"input": step_input, "step_outputs": step_outputs})

            call = await gateway.call(
                model_id=raw_model_id,
                task_name=task.name,
                sample=sample,
                sample_cache_id=f"{sample_id}:{step.id}:{task.name}",
                messages=messages,
                params_override=merged_params,
            )

            parsed = task.parse(call.content) if call.success else None
            parse_success = parsed is not None
            task_metrics = task.metrics(sample, call.content, parsed)
            spec = registry.get(raw_model_id)
            # P0 Fix: 缓存命中不产生 API 调用，成本应为 0
            cost = 0.0 if call.from_cache else estimate_cost(call.usage, spec.price_config)

            e2e_success = e2e_success and call.success and (parse_success or not task.parse_applicable)
            e2e_latency += call.latency_ms
            e2e_tokens += call.usage["total_tokens"]
            e2e_retry += call.retry_count
            e2e_cost += cost
            # e2e parsed 始终反映“最后执行步骤”的解析结果，避免被上游步骤污染。
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
                "success": call.success,
                "error": call.error,
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


async def run(args: argparse.Namespace) -> Path:
    """主执行函数：完成评测、聚合与报告生成。"""
    if bool(args.task) == bool(args.workflow):
        raise ValueError("Exactly one of --task or --workflow must be provided")

    store = RunStore(args.out)
    run_id = store.run_id
    dataset = _load_dataset(args.dataset, max_samples=args.max_samples)
    registry = ModelRegistry(args.model_registry)
    global_params = _parse_params(args.params)
    cache = None if args.no_cache else EvalCache(args.cache_dir)
    gateway = LLMGateway(registry=registry, cache=cache, mock=args.mock)

    config_payload = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(args.dataset),
        "task": args.task,
        "workflow": args.workflow,
        "models": args.models,
        "params": global_params,
        "repeats": args.repeats,
        "model_registry": str(args.model_registry),
        "git_commit": _git_commit_hash(),
        "env_keys": sorted([k for k in os.environ.keys() if k.endswith("API_KEY")]),
        "scorer_version": SCORER_VERSION,
        "result_schema_version": RESULT_ROW_SCHEMA_VERSION,
        "summary_schema_version": SUMMARY_ROW_SCHEMA_VERSION,
    }
    store.write_config(config_payload)

    if args.task:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
        if not model_ids:
            model_ids = [registry.active] if registry.active else []
        if not model_ids:
            raise ValueError("No model ids supplied and no active model in registry")
        records = await _run_task_mode(
            run_id=run_id,
            store=store,
            dataset=dataset,
            registry=registry,
            task_name=args.task,
            model_ids=model_ids,
            params_override=global_params,
            repeats=max(1, int(args.repeats)),
            gateway=gateway,
            concurrency_override=getattr(args, "concurrency", None),
        )
    else:
        workflow = load_workflow(args.workflow)
        default_model_id, step_model_map = _parse_workflow_models(args.models, registry)
        records = await _run_workflow_mode(
            run_id=run_id,
            store=store,
            dataset=dataset,
            registry=registry,
            workflow=workflow,
            global_params=global_params,
            gateway=gateway,
            default_model_id=default_model_id,
            step_model_map=step_model_map,
            workflow_concurrency=max(1, int(getattr(args, "workflow_concurrency", 1))),
        )

    summary_rows = aggregate_records(records)
    store.write_summary(summary_rows)
    store.write_partitioned_results(records)
    MarkdownReporter().generate(store.run_dir)
    return store.run_dir


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="LLM benchmark runner")
    parser.add_argument("--task", help="Task name, e.g. ie_json")
    parser.add_argument("--workflow", help="Workflow YAML/JSON path")
    parser.add_argument("--dataset", required=True, help="Dataset JSONL path")
    parser.add_argument(
        "--models",
        default="",
        help=(
            "task 模式：逗号分隔的 model ID，如 gpt-4o-mini,kimi-personal；"
            "workflow 模式：单一 model ID（全部步骤使用），"
            "或 step1=modelA,step2=modelB 按步骤指定；"
            "省略时回退到 registry 中的 active 模型"
        ),
    )
    parser.add_argument("--model-registry", default="configs/llm_providers.yaml")
    parser.add_argument("--out", default="runs/")
    parser.add_argument("--params", default=None, help='JSON override, e.g. {"temperature":0.2}')
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cache-dir", default=".eval_cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=(
            "task 模式最大并发请求数；0=无限制。"
            "省略时自动从模型配置的 concurrency 字段读取（默认 8）"
        ),
    )
    parser.add_argument(
        "--workflow-concurrency",
        type=int,
        default=1,
        help=(
            "workflow 模式样本级并发数；每个样本内部仍按 step 顺序执行。"
            "默认 1，0 视为 1。"
        ),
    )
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock responses without LLM API calls")
    args = parser.parse_args()

    run_dir = asyncio.run(run(args))
    print(str(run_dir))


if __name__ == "__main__":
    main()
