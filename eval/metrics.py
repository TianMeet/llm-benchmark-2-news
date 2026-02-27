"""Metrics aggregation utilities for evaluation outputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(len(values) * 0.95) - 1))
    return float(values[idx])


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate raw result records into summary rows."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = (
            str(r.get("task_name", "")),
            str(r.get("model_id", "")),
            str(r.get("step_id", "__task__")),
        )
        grouped[key].append(r)

    rows: list[dict[str, Any]] = []
    for (task_name, model_id, step_id), items in grouped.items():
        request_count = len(items)
        success_count = sum(1 for i in items if i.get("success"))
        parse_applicable = [i for i in items if i.get("parse_applicable", True)]
        parse_success_count = sum(1 for i in parse_applicable if i.get("parse_success"))
        latencies = [float(i.get("latency_ms", 0.0)) for i in items]
        tokens = [int(i.get("total_tokens", 0)) for i in items]
        retry_count = sum(int(i.get("retry_count", 0)) for i in items)
        total_cost = sum(float(i.get("cost_estimate", 0.0)) for i in items)

        row = {
            "task_name": task_name,
            "model_id": model_id,
            "step_id": step_id,
            "request_count": request_count,
            "success_count": success_count,
            "success_rate": round(success_count / request_count, 4) if request_count else 0.0,
            "avg_latency_ms": round(sum(latencies) / request_count, 2) if request_count else 0.0,
            "p95_latency_ms": round(_p95(latencies), 2),
            "avg_total_tokens": round(sum(tokens) / request_count, 2) if request_count else 0.0,
            "p95_total_tokens": _p95([float(t) for t in tokens]),
            "parse_rate": round(parse_success_count / len(parse_applicable), 4) if parse_applicable else 0.0,
            "retry_count": retry_count,
            "cost_estimate": round(total_cost, 6),
        }
        rows.append(row)
    return sorted(rows, key=lambda x: (x["task_name"], x["step_id"], x["model_id"]))
