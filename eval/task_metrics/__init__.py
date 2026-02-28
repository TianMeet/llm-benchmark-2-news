"""任务指标计算模块：将 GenericTask 中各指标类型的计算逻辑独立为可单独测试的函数。

使用方式
-------
from eval.task_metrics import compute_metric

result = compute_metric(mtype="exact_match", rule=rule, sample=sample, parsed=parsed)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.task_metrics.calculators import (
        as_set_of_str,
        compute_metric,
        extract_text_fragments,
        normalize_value,
        safe_float,
    )

_SYMBOLS: dict[str, tuple[str, str]] = {
    "compute_metric": ("eval.task_metrics.calculators", "compute_metric"),
    "normalize_value": ("eval.task_metrics.calculators", "normalize_value"),
    "safe_float": ("eval.task_metrics.calculators", "safe_float"),
    "as_set_of_str": ("eval.task_metrics.calculators", "as_set_of_str"),
    "extract_text_fragments": ("eval.task_metrics.calculators", "extract_text_fragments"),
}


def __getattr__(name: str):  # noqa: ANN201
    if name in _SYMBOLS:
        mod_path, attr = _SYMBOLS[name]
        return getattr(importlib.import_module(mod_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "compute_metric",
    "normalize_value",
    "safe_float",
    "as_set_of_str",
    "extract_text_fragments",
]
