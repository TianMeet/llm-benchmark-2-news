"""任务注册表：扫描 eval/tasks/*.yaml 自动发现任务，无需手写 Python 类。

新增任务只需在 eval/tasks/ 下放一个同名 YAML 文件即可。
"""

from __future__ import annotations

from pathlib import Path

from eval.tasks.base import EvalTask
from eval.tasks.generic import GenericTask

_TASKS_DIR = Path(__file__).resolve().parent


def build_task(task_name: str) -> EvalTask:
    """按任务名加载对应的 YAML 配置并返回 GenericTask 实例。

    任务配置文件路径：eval/tasks/{task_name}.yaml
    新增任务无需修改此文件，只需添加对应 YAML。
    """
    config_path = _TASKS_DIR / f"{task_name}.yaml"
    if not config_path.exists():
        available = sorted(p.stem for p in _TASKS_DIR.glob("*.yaml"))
        raise ValueError(
            f"Unknown task '{task_name}'. "
            f"No config found at {config_path}. "
            f"Available: {available}"
        )
    return GenericTask(config_path)
