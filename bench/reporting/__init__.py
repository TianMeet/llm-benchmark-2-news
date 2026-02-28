"""Reporting layer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from utils.lazy_import import make_lazy_module_getattr

if TYPE_CHECKING:
    from bench.reporting.report import generate_report
    from bench.reporting.reporter import MarkdownReporter, Reporter

__all__ = ["generate_report", "Reporter", "MarkdownReporter"]

_SYMBOLS: dict[str, tuple[str, str]] = {
    "generate_report": ("bench.reporting.report", "generate_report"),
    "MarkdownReporter": ("bench.reporting.reporter", "MarkdownReporter"),
    "Reporter": ("bench.reporting.reporter", "Reporter"),
}
__getattr__ = make_lazy_module_getattr(_SYMBOLS, __name__)
