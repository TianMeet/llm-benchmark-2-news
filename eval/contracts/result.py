"""评测系统核心契约定义。"""

from __future__ import annotations

__all__ = ["UnifiedCallResult", "ResultRow"]

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UnifiedCallResult:
    """统一的模型调用返回结构，供 runner/gateway/store 共享。"""

    content: str
    usage: dict[str, int]
    latency_ms: float
    raw_response: dict[str, Any]
    success: bool
    error: str
    retry_count: int
    from_cache: bool = False


@dataclass
class ResultRow:
    """评测结果行的强类型契约。

    每条写入 results.jsonl 的记录都应满足此契约，显式定义所有字段
    可避免 runner 中散落的裸字典拼写错误，并通过 validate() 强制不变量。

    不变量（由 validate() 检查）：
    - from_cache=True 时 cost_estimate 必须为 0.0（命中缓存不产生 API 费用）
    """

    # ── 元信息 ─────────────────────────────────────────────────────────
    schema_version: str
    run_id: str
    mode: str                      # "task" | "workflow_step" | "workflow_e2e"

    # ── 样本与任务标识 ──────────────────────────────────────────────────
    sample_id: str
    task_name: str
    task_version: str
    step_id: str
    prompt_version: str
    model_id: str
    params: dict[str, Any] = field(default_factory=dict)
    repeat_idx: int = 0

    # ── 调用结果 ───────────────────────────────────────────────────────
    success: bool = False
    error: str = ""
    parse_applicable: bool = True
    parse_success: bool = False
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    retry_count: int = 0
    cost_estimate: float = 0.0
    from_cache: bool = False

    # ── 内容与指标 ─────────────────────────────────────────────────────
    output_text: str = ""
    parsed: dict[str, Any] | None = None
    task_metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """构造后自动校验不变量，防止调用方忘记手动调用 validate()。"""
        self.validate()

    def validate(self) -> None:
        """检查业务不变量，违反时抛出 ValueError。

        当前不变量：
        1. from_cache=True 时 cost_estimate 必须为 0.0
        """
        if self.from_cache and self.cost_estimate != 0.0:
            raise ValueError(
                f"ResultRow 不变量违反：from_cache=True 时 cost_estimate 必须为 0.0，"
                f"实际值为 {self.cost_estimate}（sample_id={self.sample_id!r}, "
                f"model_id={self.model_id!r}, step_id={self.step_id!r}）"
            )
