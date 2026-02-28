"""评测执行入口：支持单任务评测与多步 workflow 评测。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
logger = logging.getLogger(__name__)

from eval.execution.gateway import LLMGateway
from eval.execution.task_runner import run_task_mode
from eval.execution.utils import (
    RESULT_ROW_SCHEMA_VERSION,
    SCORER_VERSION,
    SUMMARY_ROW_SCHEMA_VERSION,
    git_commit_hash,
    load_dataset,
    parse_params,
    parse_workflow_models,
)
from eval.execution.workflow_runner import run_workflow_mode
from eval.io.cache import EvalCache
from eval.io.store import RunStore
from eval.metrics.aggregate import aggregate_records
from eval.registry import ModelRegistry
from eval.reporting.reporter import MarkdownReporter
from eval.workflow import load_workflow

__all__ = [
    "run",
    "main",
]


async def run(args: argparse.Namespace) -> Path:
    """主执行函数：完成评测、聚合与报告生成。"""
    if bool(args.task) == bool(args.workflow):
        raise ValueError("Exactly one of --task or --workflow must be provided")

    store = RunStore(args.out)
    run_id = store.run_id
    dataset = load_dataset(args.dataset, max_samples=args.max_samples)
    registry = ModelRegistry(args.model_registry)
    global_params = parse_params(args.params)
    cache = None if args.no_cache else EvalCache(args.cache_dir)
    gateway = LLMGateway(registry=registry, cache=cache, mock=args.mock)

    config_payload = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": str(args.dataset),
        "task": args.task,
        "workflow": args.workflow,
        "models": args.models,
        "params": global_params,
        "repeats": args.repeats,
        "model_registry": str(args.model_registry),
        "git_commit": git_commit_hash(),
        "env_keys": sorted([k for k in os.environ.keys() if k.endswith("API_KEY")]),
        "scorer_version": SCORER_VERSION,
        "result_schema_version": RESULT_ROW_SCHEMA_VERSION,
        "summary_schema_version": SUMMARY_ROW_SCHEMA_VERSION,
    }
    store.write_config(config_payload)

    if args.task:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
        if not model_ids:
            model_ids = [registry.active] if registry.active else []
        if not model_ids:
            raise ValueError("No model ids supplied and no active model in registry")
        records = await run_task_mode(
            run_id=run_id,
            store=store,
            dataset=dataset,
            registry=registry,
            task_name=args.task,
            model_ids=model_ids,
            params_override=global_params,
            repeats=max(1, int(args.repeats)),
            gateway=gateway,
            concurrency_override=getattr(args, "concurrency", None),
        )
    else:
        workflow = load_workflow(args.workflow)
        default_model_id, step_model_map = parse_workflow_models(args.models, registry)
        records = await run_workflow_mode(
            run_id=run_id,
            store=store,
            dataset=dataset,
            registry=registry,
            workflow=workflow,
            global_params=global_params,
            gateway=gateway,
            default_model_id=default_model_id,
            step_model_map=step_model_map,
            workflow_concurrency=max(1, int(getattr(args, "workflow_concurrency", 1))),
        )

    summary_rows = aggregate_records(records)
    store.write_summary(summary_rows)
    store.write_partitioned_results(records)
    MarkdownReporter().generate(store.run_dir)
    return store.run_dir


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="LLM benchmark runner")
    parser.add_argument("--task", help="Task name, e.g. ie_json")
    parser.add_argument("--workflow", help="Workflow YAML/JSON path")
    parser.add_argument("--dataset", required=True, help="Dataset JSONL path")
    parser.add_argument(
        "--models",
        default="",
        help=(
            "task 模式：逗号分隔的 model ID，如 gpt-4o-mini,kimi-personal；"
            "workflow 模式：单一 model ID（全部步骤使用），"
            "或 step1=modelA,step2=modelB 按步骤指定；"
            "省略时回退到 registry 中的 active 模型"
        ),
    )
    parser.add_argument("--model-registry", default="configs/llm_providers.yaml")
    parser.add_argument("--out", default="runs/")
    parser.add_argument("--params", default=None, help='JSON override, e.g. {"temperature":0.2}')
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cache-dir", default=".eval_cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=(
            "task 模式最大并发请求数；0=无限制。"
            "省略时自动从模型配置的 concurrency 字段读取（默认 8）"
        ),
    )
    parser.add_argument(
        "--workflow-concurrency",
        type=int,
        default=1,
        help=(
            "workflow 模式样本级并发数；每个样本内部仍按 step 顺序执行。"
            "默认 1，0 视为 1。"
        ),
    )
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock responses without LLM API calls")
    args = parser.parse_args()

    run_dir = asyncio.run(run(args))
    print(str(run_dir))


if __name__ == "__main__":
    main()
