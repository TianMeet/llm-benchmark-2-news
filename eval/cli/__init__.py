"""CLI entrypoints for evaluation workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.cli.runner import main, run

__all__ = ["run", "main"]


def __getattr__(name: str):
    if name in ("run", "main"):
        from eval.cli import runner  # noqa: PLC0415

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
