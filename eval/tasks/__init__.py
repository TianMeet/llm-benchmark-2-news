"""Task registry for benchmark tasks."""

from eval.tasks.base import EvalTask
from eval.tasks.ie_json import IEJsonTask
from eval.tasks.stock_score import StockScoreTask
from eval.tasks.news_dedup import NewsDedupTask


def build_task(task_name: str) -> EvalTask:
    tasks = {
        "ie_json": IEJsonTask(),
        "stock_score": StockScoreTask(),
        "news_dedup": NewsDedupTask(),
    }
    if task_name not in tasks:
        raise ValueError(f"Unknown task '{task_name}'. Available: {sorted(tasks.keys())}")
    return tasks[task_name]
