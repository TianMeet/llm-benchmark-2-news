"""核心数据契约层 — 跨模块共享的类型定义与错误分类。

本包使用惰性导入(lazy import)，避免 import bench.contracts 时
提前触发所有子模块的加载。实际使用时按需导入：
    from bench.contracts import ResultRow, ScoreCard, derive_error_fields

子模块：
- result.py    : UnifiedCallResult / ResultRow
- scoring.py   : CanonicalAnswer / ScoreCard / ScoringProfile 等评分模型
- error.py     : classify_error / derive_error_fields（结构化错误归一化）
- exceptions.py: EvalBenchError 异常层级
- constants.py : STEP_ID_TASK / STEP_ID_E2E
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from bench_utils.lazy_import import make_lazy_module_getattr

if TYPE_CHECKING:
    from bench.contracts.constants import STEP_ID_E2E, STEP_ID_TASK
    from bench.contracts.error import classify_error, derive_error_fields
    from bench.contracts.exceptions import EvalBenchError
    from bench.contracts.result import ResultRow, UnifiedCallResult
    from bench.contracts.scoring import (
        AnswerType,
        CanonicalAnswer,
        MetricRule,
        NormalizationConfig,
        ScoreCard,
        ScorerPreset,
        ScoringProfile,
    )

__all__ = [
    "UnifiedCallResult",
    "ResultRow",
    "classify_error",
    "derive_error_fields",
    "STEP_ID_TASK",
    "STEP_ID_E2E",
    "EvalBenchError",
    # scoring contract models
    "AnswerType",
    "CanonicalAnswer",
    "ScoreCard",
    "MetricRule",
    "ScorerPreset",
    "NormalizationConfig",
    "ScoringProfile",
]

_SYMBOLS: dict[str, tuple[str, str]] = {
    "ResultRow": ("bench.contracts.result", "ResultRow"),
    "UnifiedCallResult": ("bench.contracts.result", "UnifiedCallResult"),
    "classify_error": ("bench.contracts.error", "classify_error"),
    "derive_error_fields": ("bench.contracts.error", "derive_error_fields"),
    "STEP_ID_TASK": ("bench.contracts.constants", "STEP_ID_TASK"),
    "STEP_ID_E2E": ("bench.contracts.constants", "STEP_ID_E2E"),
    "EvalBenchError": ("bench.contracts.exceptions", "EvalBenchError"),
    # scoring contract models
    "AnswerType": ("bench.contracts.scoring", "AnswerType"),
    "CanonicalAnswer": ("bench.contracts.scoring", "CanonicalAnswer"),
    "ScoreCard": ("bench.contracts.scoring", "ScoreCard"),
    "MetricRule": ("bench.contracts.scoring", "MetricRule"),
    "ScorerPreset": ("bench.contracts.scoring", "ScorerPreset"),
    "NormalizationConfig": ("bench.contracts.scoring", "NormalizationConfig"),
    "ScoringProfile": ("bench.contracts.scoring", "ScoringProfile"),
}
__getattr__ = make_lazy_module_getattr(_SYMBOLS, __name__)
