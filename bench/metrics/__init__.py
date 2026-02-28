"""评测指标聚合层。

将 results.jsonl 明细记录聚合为 summary.csv 汇总行。
主入口：aggregate_records(records) -> list[dict]
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from bench_utils.lazy_import import make_lazy_module_getattr

if TYPE_CHECKING:
    from bench.metrics.aggregate import SUMMARY_ROW_SCHEMA_VERSION, aggregate_records

__all__ = ["aggregate_records", "SUMMARY_ROW_SCHEMA_VERSION"]

_SYMBOLS: dict[str, tuple[str, str]] = {
    "aggregate_records": ("bench.metrics.aggregate", "aggregate_records"),
    "SUMMARY_ROW_SCHEMA_VERSION": ("bench.metrics.aggregate", "SUMMARY_ROW_SCHEMA_VERSION"),
}
__getattr__ = make_lazy_module_getattr(_SYMBOLS, __name__)
