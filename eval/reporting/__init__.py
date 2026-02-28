"""Reporting layer."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.reporting.report import generate_report
    from eval.reporting.reporter import MarkdownReporter, Reporter

__all__ = ["generate_report", "Reporter", "MarkdownReporter"]

_SYMBOLS: dict[str, tuple[str, str]] = {
    "generate_report": ("eval.reporting.report", "generate_report"),
    "MarkdownReporter": ("eval.reporting.reporter", "MarkdownReporter"),
    "Reporter": ("eval.reporting.reporter", "Reporter"),
}


def __getattr__(name: str):  # noqa: ANN201
    if name in _SYMBOLS:
        mod_path, attr = _SYMBOLS[name]
        return getattr(importlib.import_module(mod_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
