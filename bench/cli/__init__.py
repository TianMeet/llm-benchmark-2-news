"""CLI entrypoints for evaluation workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING
from bench_utils.lazy_import import make_lazy_module_getattr

if TYPE_CHECKING:
    from bench.cli.runner import main, run

__all__ = ["run", "main"]

_SYMBOLS: dict[str, tuple[str, str]] = {
    "run": ("bench.cli.runner", "run"),
    "main": ("bench.cli.runner", "main"),
}

__getattr__ = make_lazy_module_getattr(_SYMBOLS, __name__)
