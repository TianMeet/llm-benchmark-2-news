"""I/O layer: persistent store, cache, and prompt loading."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.io.cache import EvalCache
    from eval.io.prompt_store import PromptStore
    from eval.io.store import RunStore

__all__ = ["EvalCache", "RunStore", "PromptStore"]

_SYMBOLS: dict[str, tuple[str, str]] = {
    "EvalCache": ("eval.io.cache", "EvalCache"),
    "PromptStore": ("eval.io.prompt_store", "PromptStore"),
    "RunStore": ("eval.io.store", "RunStore"),
}


def __getattr__(name: str):  # noqa: ANN201
    if name in _SYMBOLS:
        mod_path, attr = _SYMBOLS[name]
        return getattr(importlib.import_module(mod_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
