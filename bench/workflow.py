"""轻量 Workflow 结构定义与加载器。

支持多步任务编排，如 workflows/news_pipeline.yaml 中的：
  step1_dedup(news_dedup) → step2_extract(ie_json) → step3_score(stock_score)

核心特性：
- 步骤间依赖通过 input_from 字段声明（"sample" 或上游 step id）
- 加载时执行严格校验：step id 去重、input_from 引用校验、依赖图环检测 (DFS)
- model_id 支持 $active 占位符，运行时由 CLI --models 解析
- JSON / YAML 双格式支持

值对象设计：WorkflowStep 和 WorkflowSpec 均为 frozen dataclass，
加载后不可变，保证执行期间安全。
"""

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
    step_ids: set[str] = set()
    for s in raw.get("steps", []):
        sid = str(s["id"])
        if sid in step_ids:
            raise ValueError(f"Duplicate step id: '{sid}'")
        step_ids.add(sid)
        steps.append(
            WorkflowStep(
                id=sid,
                task=str(s["task"]),
                model_id=str(s.get("model_id", "$active")),
                input_from=str(s.get("input_from", "sample")),
                params_override=dict(s.get("params_override", {}) or {}),
            )
        )
    if not steps:
        raise ValueError("Workflow has no steps")

    # 配置前置校验：input_from 必须是 sample 或已有 step id。
    for step in steps:
        if step.input_from != "sample" and step.input_from not in step_ids:
            raise ValueError(
                f"Step '{step.id}' has unknown input_from '{step.input_from}'. "
                "Expected 'sample' or an existing step id."
            )

    # 配置前置校验：依赖图中不得存在环。
    deps = {
        step.id: (step.input_from if step.input_from != "sample" else None)
        for step in steps
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def _dfs(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ValueError(f"Workflow has cycle dependency involving step '{node}'")
        visiting.add(node)
        dep = deps.get(node)
        if dep is not None:
            _dfs(dep)
        visiting.remove(node)
        visited.add(node)

    for sid in deps:
        _dfs(sid)

    return WorkflowSpec(
        name=str(raw.get("name", file_path.stem)),
        version=str(raw.get("version", "v1")),
        steps=tuple(steps),
    )
