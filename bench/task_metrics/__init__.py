"""任务指标计算模块：将 GenericTask 中各指标类型的计算逻辑独立为可单独测试的函数。

子模块：
- calculators.py : 8 种指标计算器 + compute_metric() 统一分发
- profile.py     : ScoringProfile 加载 + resolve_metrics() 三层合并
- scoring.py     : build_score_card() 将散装 dict 包装为 ScoreCard
- regression.py  : check_regression() 回归门禁判断

支持的指标类型：
  field_completeness / reference_rouge / custom_ratio / exact_match
  numeric_error / list_overlap / reasoning_consistency / claim_evidence_overlap

使用方式：
    from bench.task_metrics import compute_metric
    result = compute_metric(mtype="exact_match", rule=rule, sample=sample, parsed=parsed)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from utils.lazy_import import make_lazy_module_getattr

if TYPE_CHECKING:
    from bench.task_metrics.calculators import (
        as_set_of_str,
        compute_metric,
        extract_text_fragments,
        normalize_value,
        safe_float,
    )

_SYMBOLS: dict[str, tuple[str, str]] = {
    "compute_metric": ("bench.task_metrics.calculators", "compute_metric"),
    "normalize_value": ("bench.task_metrics.calculators", "normalize_value"),
    "safe_float": ("bench.task_metrics.calculators", "safe_float"),
    "as_set_of_str": ("bench.task_metrics.calculators", "as_set_of_str"),
    "extract_text_fragments": ("bench.task_metrics.calculators", "extract_text_fragments"),
}
__getattr__ = make_lazy_module_getattr(_SYMBOLS, __name__)

__all__ = [
    "compute_metric",
    "normalize_value",
    "safe_float",
    "as_set_of_str",
    "extract_text_fragments",
]
