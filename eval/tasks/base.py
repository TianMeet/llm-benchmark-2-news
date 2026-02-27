"""Task contract for evaluation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EvalTask(ABC):
    """Task contract: build prompt, parse output, compute metrics."""

    name: str
    version: str
    parse_applicable: bool = True

    @abstractmethod
    def build_prompt(self, sample: dict[str, Any], context: dict[str, Any] | None = None) -> tuple[list[dict], str]:
        """Build LLM input messages and return prompt_version."""

    @abstractmethod
    def parse(self, output: str) -> dict[str, Any] | None:
        """Parse raw model output into structured result."""

    @abstractmethod
    def metrics(self, sample: dict[str, Any], output: str, parsed: dict[str, Any] | None) -> dict[str, Any]:
        """Compute task-specific metrics."""
