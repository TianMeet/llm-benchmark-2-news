"""单任务多模型评测执行器。"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from eval.execution.gateway import LLMGateway
from eval.execution.utils import RESULT_ROW_SCHEMA_VERSION, resolve_concurrency
from eval.io.store import RunStore
from eval.registry import ModelRegistry, estimate_cost
from eval.tasks import build_task

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
) -> list[dict[str, Any]]:
    """执行"单任务 + 多模型"评测模式，每个模型使用独立 Semaphore 并发。"""
    task = build_task(task_name)
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

        def _error_row(error: str) -> dict[str, Any]:
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
                "success": False,
                "error": error,
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

        async with sems[model_id]:
            try:
                if gateway.mock:
                    messages, prompt_version = [], "mock"
                else:
                    messages, prompt_version = task.build_prompt(sample, context={"repeat_idx": repeat_idx})
            except Exception as exc:
                return _error_row(f"build_prompt_error: {exc}")
            try:
                call = await gateway.call(
                    model_id=model_id,
                    task_name=task.name,
                    sample=sample,
                    sample_cache_id=f"{sample_id}:{repeat_idx}:{task.name}",
                    messages=messages,
                    params_override=params_override,
                )
            except Exception as exc:
                return _error_row(f"gateway_error: {exc}")

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
                error_message = f"{error_message}; parse_error: {exc}" if error_message else f"parse_error: {exc}"
        else:
            parse_success = False

        try:
            task_metrics = task.metrics(sample, call.content, parsed)
        except Exception as exc:
            row_success = False
            error_message = f"{error_message}; metrics_error: {exc}" if error_message else f"metrics_error: {exc}"
            task_metrics = {}

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
