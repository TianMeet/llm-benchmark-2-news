"""Core data contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING
from utils.lazy_import import make_lazy_module_getattr

if TYPE_CHECKING:
    from bench.contracts.constants import STEP_ID_E2E, STEP_ID_TASK
    from bench.contracts.error import classify_error, derive_error_fields
    from bench.contracts.exceptions import EvalBenchError
    from bench.contracts.result import ResultRow, UnifiedCallResult

__all__ = [
    "UnifiedCallResult",
    "ResultRow",
    "classify_error",
    "derive_error_fields",
    "STEP_ID_TASK",
    "STEP_ID_E2E",
    "EvalBenchError",
]

_SYMBOLS: dict[str, tuple[str, str]] = {
    "ResultRow": ("bench.contracts.result", "ResultRow"),
    "UnifiedCallResult": ("bench.contracts.result", "UnifiedCallResult"),
    "classify_error": ("bench.contracts.error", "classify_error"),
    "derive_error_fields": ("bench.contracts.error", "derive_error_fields"),
    "STEP_ID_TASK": ("bench.contracts.constants", "STEP_ID_TASK"),
    "STEP_ID_E2E": ("bench.contracts.constants", "STEP_ID_E2E"),
    "EvalBenchError": ("bench.contracts.exceptions", "EvalBenchError"),
}
__getattr__ = make_lazy_module_getattr(_SYMBOLS, __name__)
