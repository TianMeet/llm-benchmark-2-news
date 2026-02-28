"""Core data contracts."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.contracts.result import ResultRow, UnifiedCallResult

__all__ = ["UnifiedCallResult", "ResultRow"]

_SYMBOLS: dict[str, tuple[str, str]] = {
    "ResultRow": ("eval.contracts.result", "ResultRow"),
    "UnifiedCallResult": ("eval.contracts.result", "UnifiedCallResult"),
}


def __getattr__(name: str):  # noqa: ANN201
    if name in _SYMBOLS:
        mod_path, attr = _SYMBOLS[name]
        return getattr(importlib.import_module(mod_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
