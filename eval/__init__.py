"""量化新闻 LLM 评测中台（MVP）核心包。"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.contracts.result import ResultRow, UnifiedCallResult
    from eval.execution.gateway import LLMGateway, ModelGateway
    from eval.io.cache import EvalCache
    from eval.io.store import RunStore
    from eval.metrics.aggregate import aggregate_records
    from eval.registry import ModelRegistry, ModelSpec, estimate_cost
    from eval.workflow import WorkflowSpec, WorkflowStep, load_workflow

__all__ = [
    "EvalCache",
    "UnifiedCallResult",
    "ResultRow",
    "ModelGateway",
    "LLMGateway",
    "aggregate_records",
    "ModelSpec",
    "ModelRegistry",
    "estimate_cost",
    "RunStore",
    "WorkflowStep",
    "WorkflowSpec",
    "load_workflow",
]

_SYMBOLS: dict[str, tuple[str, str]] = {
    "ResultRow": ("eval.contracts.result", "ResultRow"),
    "UnifiedCallResult": ("eval.contracts.result", "UnifiedCallResult"),
    "LLMGateway": ("eval.execution.gateway", "LLMGateway"),
    "ModelGateway": ("eval.execution.gateway", "ModelGateway"),
    "EvalCache": ("eval.io.cache", "EvalCache"),
    "RunStore": ("eval.io.store", "RunStore"),
    "aggregate_records": ("eval.metrics.aggregate", "aggregate_records"),
    "ModelRegistry": ("eval.registry", "ModelRegistry"),
    "ModelSpec": ("eval.registry", "ModelSpec"),
    "estimate_cost": ("eval.registry", "estimate_cost"),
    "WorkflowSpec": ("eval.workflow", "WorkflowSpec"),
    "WorkflowStep": ("eval.workflow", "WorkflowStep"),
    "load_workflow": ("eval.workflow", "load_workflow"),
}


def __getattr__(name: str):  # noqa: ANN201
    if name in _SYMBOLS:
        mod_path, attr = _SYMBOLS[name]
        return getattr(importlib.import_module(mod_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
