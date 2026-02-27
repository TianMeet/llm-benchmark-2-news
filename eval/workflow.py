"""Lightweight workflow spec and loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WorkflowStep:
    id: str
    task: str
    model_id: str
    input_from: str = "sample"
    params_override: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowSpec:
    name: str
    version: str
    steps: list[WorkflowStep]


def load_workflow(path: str | Path) -> WorkflowSpec:
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        raw = json.loads(raw_text)
    else:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyYAML is required to load YAML workflow files. "
                "Install with: pip install pyyaml, or use JSON workflow."
            ) from exc
        raw = yaml.safe_load(raw_text)

    steps: list[WorkflowStep] = []
    for s in raw.get("steps", []):
        steps.append(
            WorkflowStep(
                id=str(s["id"]),
                task=str(s["task"]),
                model_id=str(s["model_id"]),
                input_from=str(s.get("input_from", "sample")),
                params_override=dict(s.get("params_override", {}) or {}),
            )
        )
    if not steps:
        raise ValueError("Workflow has no steps")

    return WorkflowSpec(
        name=str(raw.get("name", file_path.stem)),
        version=str(raw.get("version", "v1")),
        steps=steps,
    )
