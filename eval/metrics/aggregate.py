"""评测指标聚合工具。"""

from __future__ import annotations

__all__ = ["aggregate_records", "SUMMARY_ROW_SCHEMA_VERSION"]

import math
from collections import defaultdict
from typing import Any

SUMMARY_ROW_SCHEMA_VERSION = "summary_row.v1"


def _p95(values: list[float]) -> float | None:
    """计算 p95 分位值。

    - 空数组返回 None（调用方可区分"无数据"与"P95=0"两种语义）。
    - 使用 ceil(n * 0.95) 索引，修正原 int(n * 0.95)-1 在小数组上的偏低问题。
      例：n=5 时旧公式得 index=3（60th pct），新公式得 index=4（真正最大值，近似 P95）。
    """
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, max(0, math.ceil(len(sorted_vals) * 0.95) - 1))
    return float(sorted_vals[idx])


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将明细记录聚合为 summary 行（task/model/step 维度）。"""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        # 聚合主键：任务 + 模型 + 步骤。
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

        # 统一输出核心指标，便于横向/纵向对比。
        p95_lat = _p95(latencies)
        p95_tok = _p95([float(t) for t in tokens])
        row = {
            "schema_version": SUMMARY_ROW_SCHEMA_VERSION,
            "task_name": task_name,
            "model_id": model_id,
            "step_id": step_id,
            "request_count": request_count,
            "success_count": success_count,
            "success_rate": round(success_count / request_count, 4) if request_count else 0.0,
            "avg_latency_ms": round(sum(latencies) / request_count, 2) if request_count else 0.0,
            "p95_latency_ms": round(p95_lat, 2) if p95_lat is not None else None,
            "avg_total_tokens": round(sum(tokens) / request_count, 2) if request_count else 0.0,
            "p95_total_tokens": round(p95_tok, 2) if p95_tok is not None else None,
            "parse_rate": round(parse_success_count / len(parse_applicable), 4) if parse_applicable else 0.0,
            "retry_count": retry_count,
            "cost_estimate": round(total_cost, 6),
        }
        # 聚合任务特定指标（task_metrics 内数值字段，以 tm_ 前缀区分）。
        all_metrics_keys: set[str] = set()
        for i in items:
            all_metrics_keys.update((i.get("task_metrics") or {}).keys())
        for mk in sorted(all_metrics_keys):
            vals = [
                float(i["task_metrics"][mk])
                for i in items
                if isinstance((i.get("task_metrics") or {}).get(mk), (int, float))
            ]
            row[f"tm_{mk}_avg"] = round(sum(vals) / len(vals), 4) if vals else None
        rows.append(row)
    return sorted(rows, key=lambda x: (x["task_name"], x["step_id"], x["model_id"]))
