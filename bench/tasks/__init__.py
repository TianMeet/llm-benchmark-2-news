"""任务注册表：扫描 bench/tasks/*.yaml 自动发现任务，无需手写 Python 类。

新增任务只需在 bench/tasks/ 下放一个同名 YAML 文件即可。
所有 YAML 任务都由 GenericTask 统一实例化，无需修改任何 Python 代码。

当前已注册任务：
- ie_json             : 新闻信息抽取（股票代码、事件类型、情感、影响度、摘要）
- news_analysis       : 新闻摘要 + 情感分析（支持 gt_sentiment/reference_summary 评测）
- news_dedup          : 新闻去重（去重标题列表、移除数量）
- stock_score         : 股票评分（-100~100 分值 + 理由）
- stock_association   : 股票联想（板块→关联股票列表，precision/recall/F1 评测）
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bench.tasks.base import EvalTask
    from bench.tasks.generic import GenericTask

_TASKS_DIR = Path(__file__).resolve().parent


def build_task(task_name: str) -> "EvalTask":
    """按任务名加载对应的 YAML 配置并返回 GenericTask 实例。

    任务配置文件路径：bench/tasks/{task_name}.yaml
    新增任务无需修改此文件，只需添加对应 YAML。
    """
    from bench.tasks.generic import GenericTask  # noqa: PLC0415

    config_path = _TASKS_DIR / f"{task_name}.yaml"
    if not config_path.exists():
        available = sorted(p.stem for p in _TASKS_DIR.glob("*.yaml"))
        raise ValueError(
            f"Unknown task '{task_name}'. "
            f"No config found at {config_path}. "
            f"Available: {available}"
        )
    return GenericTask(config_path)
