"""任务指标计算模块：将 GenericTask 中各指标类型的计算逻辑独立为可单独测试的函数。

使用方式
-------
from eval.task_metrics import compute_metric

result = compute_metric(mtype="exact_match", rule=rule, sample=sample, parsed=parsed)
"""

from __future__ import annotations

from eval.task_metrics.calculators import (
    compute_metric,
    normalize_value,
    safe_float,
    as_set_of_str,
    extract_text_fragments,
)

__all__ = [
    "compute_metric",
    "normalize_value",
    "safe_float",
    "as_set_of_str",
    "extract_text_fragments",
]
