"""Execution layer: task/workflow runners and gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING
from utils.lazy_import import make_lazy_module_getattr

if TYPE_CHECKING:
    from bench.execution.gateway import LLMGateway, ModelGateway
    from bench.execution.task_runner import run_task_mode
    from bench.execution.workflow_runner import run_workflow_mode

__all__ = ["ModelGateway", "LLMGateway", "run_task_mode", "run_workflow_mode"]

_SYMBOLS: dict[str, tuple[str, str]] = {
    "LLMGateway": ("bench.execution.gateway", "LLMGateway"),
    "ModelGateway": ("bench.execution.gateway", "ModelGateway"),
    "run_task_mode": ("bench.execution.task_runner", "run_task_mode"),
    "run_workflow_mode": ("bench.execution.workflow_runner", "run_workflow_mode"),
}
__getattr__ = make_lazy_module_getattr(_SYMBOLS, __name__)
