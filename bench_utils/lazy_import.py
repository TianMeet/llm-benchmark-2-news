"""惰性导入辅助工具 — 用于包级 __getattr__ 钩子。

被 bench/ 和 llm_core/ 的各 __init__.py 使用，
实现 import bench.xxx 时只导入实际访问的符号，
避免拉入全部子模块（降低启动开销和循环引用风险）。
"""

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
