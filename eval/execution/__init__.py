"""Execution layer: task/workflow runners and gateway."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.execution.gateway import LLMGateway, ModelGateway
    from eval.execution.task_runner import run_task_mode
    from eval.execution.workflow_runner import run_workflow_mode

__all__ = ["ModelGateway", "LLMGateway", "run_task_mode", "run_workflow_mode"]

_SYMBOLS: dict[str, tuple[str, str]] = {
    "LLMGateway": ("eval.execution.gateway", "LLMGateway"),
    "ModelGateway": ("eval.execution.gateway", "ModelGateway"),
    "run_task_mode": ("eval.execution.task_runner", "run_task_mode"),
    "run_workflow_mode": ("eval.execution.workflow_runner", "run_workflow_mode"),
}


def __getattr__(name: str):  # noqa: ANN201
    if name in _SYMBOLS:
        mod_path, attr = _SYMBOLS[name]
        return getattr(importlib.import_module(mod_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
