"""Shared lazy import helpers for package-level __getattr__ hooks."""

from __future__ import annotations

import importlib
from collections.abc import Callable

__all__ = ["make_lazy_module_getattr"]


def make_lazy_module_getattr(
    symbols: dict[str, tuple[str, str]],
    module_name: str,
) -> Callable[[str], object]:
    """Build a standard __getattr__ implementation from symbol mapping."""

    def _lazy_getattr(name: str) -> object:
        if name in symbols:
            mod_path, attr = symbols[name]
            return getattr(importlib.import_module(mod_path), attr)
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return _lazy_getattr
