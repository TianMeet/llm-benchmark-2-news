"""Evaluation metric aggregation."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.metrics.aggregate import SUMMARY_ROW_SCHEMA_VERSION, aggregate_records

__all__ = ["aggregate_records", "SUMMARY_ROW_SCHEMA_VERSION"]

_SYMBOLS: dict[str, tuple[str, str]] = {
    "aggregate_records": ("eval.metrics.aggregate", "aggregate_records"),
    "SUMMARY_ROW_SCHEMA_VERSION": ("eval.metrics.aggregate", "SUMMARY_ROW_SCHEMA_VERSION"),
}


def __getattr__(name: str):  # noqa: ANN201
    if name in _SYMBOLS:
        mod_path, attr = _SYMBOLS[name]
        return getattr(importlib.import_module(mod_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
