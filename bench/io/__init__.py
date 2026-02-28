"""
I/O 层：运行产物持久化、缓存、Prompt 加载。

模块概览：
- store.py        : RunStore — 每次评测运行的全部文件产物管理
- cache.py        : EvalCache — 基于 JSON 文件的持久化缓存
- prompt_store.py : PromptStore — Jinja2 Prompt 模板懒加载
"""

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
