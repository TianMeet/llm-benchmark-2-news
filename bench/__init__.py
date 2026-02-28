"""量化新闻 LLM 评测中台（MVP）核心包。

包结构概览：
  bench/
  ├── contracts/    → 跨模块数据契约（ResultRow / ScoreCard / 异常层级）
  ├── tasks/        → YAML 驱动的任务定义（GenericTask）
  ├── execution/    → 调用网关 + task/workflow 执行器
  ├── io/           → 运行产物存储、缓存、Prompt 加载
  ├── metrics/      → 聚合统计
  ├── task_metrics/ → 8 种指标计算器 + ScoringProfile
  ├── reporting/    → Markdown 报告生成
  ├── registry.py   → 模型注册表 + 成本估算
  ├── workflow.py   → Workflow YAML/JSON 加载与校验
  └── cli/runner.py → CLI 执行入口

使用惰性导入（lazy import）避免 import bench 时拉入全部子模块。
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from utils.lazy_import import make_lazy_module_getattr

if TYPE_CHECKING:
    from bench.contracts.constants import STEP_ID_E2E, STEP_ID_TASK
    from bench.contracts.error import classify_error, derive_error_fields
    from bench.contracts.result import ResultRow, UnifiedCallResult
    from bench.execution.gateway import LLMGateway, ModelGateway
    from bench.io.cache import EvalCache
    from bench.io.store import RunStore
    from bench.metrics.aggregate import aggregate_records
    from bench.registry import ModelRegistry, ModelSpec, estimate_cost
    from bench.workflow import WorkflowSpec, WorkflowStep, load_workflow

__all__ = [
    "EvalCache",
    "UnifiedCallResult",
    "ResultRow",
    "classify_error",
    "derive_error_fields",
    "STEP_ID_TASK",
    "STEP_ID_E2E",
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
    "ResultRow": ("bench.contracts.result", "ResultRow"),
    "UnifiedCallResult": ("bench.contracts.result", "UnifiedCallResult"),
    "classify_error": ("bench.contracts.error", "classify_error"),
    "derive_error_fields": ("bench.contracts.error", "derive_error_fields"),
    "STEP_ID_TASK": ("bench.contracts.constants", "STEP_ID_TASK"),
    "STEP_ID_E2E": ("bench.contracts.constants", "STEP_ID_E2E"),
    "LLMGateway": ("bench.execution.gateway", "LLMGateway"),
    "ModelGateway": ("bench.execution.gateway", "ModelGateway"),
    "EvalCache": ("bench.io.cache", "EvalCache"),
    "RunStore": ("bench.io.store", "RunStore"),
    "aggregate_records": ("bench.metrics.aggregate", "aggregate_records"),
    "ModelRegistry": ("bench.registry", "ModelRegistry"),
    "ModelSpec": ("bench.registry", "ModelSpec"),
    "estimate_cost": ("bench.registry", "estimate_cost"),
    "WorkflowSpec": ("bench.workflow", "WorkflowSpec"),
    "WorkflowStep": ("bench.workflow", "WorkflowStep"),
    "load_workflow": ("bench.workflow", "load_workflow"),
}
__getattr__ = make_lazy_module_getattr(_SYMBOLS, __name__)
