"""轻量 workflow 结构定义与加载器。"""

from __future__ import annotations

__all__ = ["WorkflowStep", "WorkflowSpec", "load_workflow"]

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkflowStep:
    """单个工作流步骤定义（不可变值对象）。"""
    id: str
    task: str
    model_id: str = "$active"   # 省略或设为 "$active" 时由运行时解析
    input_from: str = "sample"
    params_override: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowSpec:
    """工作流整体定义（不可变值对象）。"""
    name: str
    version: str
    steps: tuple[WorkflowStep, ...]


def load_workflow(path: str | Path) -> WorkflowSpec:
    """从 JSON/YAML 加载 workflow，并转换为强类型结构。"""
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        raw = json.loads(raw_text)
    else:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            # YAML 依赖缺失时给出明确安装提示。
            raise ModuleNotFoundError(
                "PyYAML is required to load YAML workflow files. "
                "Install with: pip install pyyaml, or use JSON workflow."
            ) from exc
        raw = yaml.safe_load(raw_text)

    if not isinstance(raw, dict):
        raise ValueError(f"Workflow root must be an object: {file_path}")

    steps: list[WorkflowStep] = []
    for s in raw.get("steps", []):
        steps.append(
            WorkflowStep(
                id=str(s["id"]),
                task=str(s["task"]),
                model_id=str(s.get("model_id", "$active")),
                input_from=str(s.get("input_from", "sample")),
                params_override=dict(s.get("params_override", {}) or {}),
            )
        )
    if not steps:
        raise ValueError("Workflow has no steps")

    return WorkflowSpec(
        name=str(raw.get("name", file_path.stem)),
        version=str(raw.get("version", "v1")),
        steps=tuple(steps),
    )
