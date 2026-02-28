"""I/O layer: persistent store, cache, and prompt loading."""

from __future__ import annotations

from typing import TYPE_CHECKING
from utils.lazy_import import make_lazy_module_getattr

if TYPE_CHECKING:
    from bench.io.cache import EvalCache
    from bench.io.prompt_store import PromptStore
    from bench.io.store import RunStore

__all__ = ["EvalCache", "RunStore", "PromptStore"]

_SYMBOLS: dict[str, tuple[str, str]] = {
    "EvalCache": ("bench.io.cache", "EvalCache"),
    "PromptStore": ("bench.io.prompt_store", "PromptStore"),
    "RunStore": ("bench.io.store", "RunStore"),
}
__getattr__ = make_lazy_module_getattr(_SYMBOLS, __name__)
