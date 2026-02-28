"""跨模块共享常量 — 结果行 step_id 标识符。

step_id 用于区分结果行属于普通任务执行还是 workflow 端到端汇总：
- STEP_ID_TASK  ("__task__")：单任务模式或 workflow 中的独立步骤。
- STEP_ID_E2E   ("__e2e__")：workflow 模式下的端到端汇总行，
  汇总该样本所有步骤的耗时/token/成本，用于整体效果回归。
"""

from __future__ import annotations

__all__ = ["STEP_ID_TASK", "STEP_ID_E2E"]

STEP_ID_TASK = "__task__"   # 单任务评测或 workflow step 级结果行
STEP_ID_E2E = "__e2e__"     # workflow 端到端汇总行
