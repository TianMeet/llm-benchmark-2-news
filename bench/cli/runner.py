"""评测执行入口：支持单任务评测与多步 workflow 评测。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

from bench.execution.gateway import LLMGateway
from bench.execution.task_runner import run_task_mode
from bench.execution.utils import (
    RESULT_ROW_SCHEMA_VERSION,
    SCORER_VERSION,
    SUMMARY_ROW_SCHEMA_VERSION,
    dataset_fingerprint,
    git_commit_hash,
    git_is_dirty,
    load_dataset,
    parse_params,
    parse_workflow_models,
    redact_sensitive_fields,
    runtime_metadata,
    sha256_file,
)
from bench.contracts.constants import STEP_ID_E2E, STEP_ID_TASK
from bench.execution.workflow_runner import run_workflow_mode
from bench.io.cache import EvalCache
from bench.io.store import RunStore
from bench.metrics.aggregate import aggregate_records
from bench.registry import ModelRegistry
from bench.reporting.reporter import MarkdownReporter
from bench.workflow import load_workflow

__all__ = [
    "run",
    "main",
]


async def run(args: argparse.Namespace) -> Path:
    """主执行函数：完成评测、聚合与报告生成。"""
    if bool(args.task) == bool(args.workflow):
        raise ValueError("Exactly one of --task or --workflow must be provided")

    if getattr(args, "resume_run_dir", None):
        store = RunStore(args.out, existing_run_dir=args.resume_run_dir)
    else:
        store = RunStore(args.out)
    run_id = store.run_id
    dataset = load_dataset(args.dataset, max_samples=args.max_samples)
    registry = ModelRegistry(args.model_registry)
    global_params = parse_params(args.params)
    cache = None if args.no_cache else EvalCache(args.cache_dir)
    gateway = LLMGateway(registry=registry, cache=cache, mock=args.mock)
    completed_keys = store.load_completed_keys() if getattr(args, "resume_run_dir", None) else set()
    workflow_spec = load_workflow(args.workflow) if args.workflow else None

    model_snapshot: dict[str, dict[str, object]] = {}
    workflow_sha = ""
    task_yaml_sha = ""
    prompt_sha = ""

    if args.task:
        task_yaml = Path(__file__).resolve().parents[1] / "tasks" / f"{args.task}.yaml"
        prompt_yaml = ""
        if task_yaml.exists():
            task_yaml_sha = sha256_file(task_yaml)
            try:
                import yaml
                raw_task_cfg = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
                prompt_name = str(raw_task_cfg.get("prompt_template", ""))
                if prompt_name:
                    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / f"{prompt_name}.yaml"
                    if prompt_path.exists():
                        prompt_yaml = str(prompt_path)
                        prompt_sha = sha256_file(prompt_path)
            except Exception:
                prompt_yaml = ""
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
        if not model_ids:
            model_ids = [registry.active] if registry.active else []
        for mid in model_ids:
            resolved = dict(registry.resolve_config(mid, global_params))
            model_snapshot[mid] = redact_sensitive_fields(resolved)
    else:
        workflow_sha = sha256_file(args.workflow)
        default_model_id, step_model_map = parse_workflow_models(args.models, registry)
        assert workflow_spec is not None
        for step in workflow_spec.steps:
            model_id = step_model_map.get(step.id) or (
                default_model_id if step.model_id == "$active" else step.model_id
            )
            if not model_id:
                continue
            merged = dict(registry.resolve_config(model_id, {**global_params, **step.params_override}))
            model_snapshot[f"{step.id}:{model_id}"] = redact_sensitive_fields(merged)

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
        "git_dirty": git_is_dirty(),
        "env_keys": sorted([k for k in os.environ.keys() if k.endswith("API_KEY")]),
        "scorer_version": SCORER_VERSION,
        "result_schema_version": RESULT_ROW_SCHEMA_VERSION,
        "summary_schema_version": SUMMARY_ROW_SCHEMA_VERSION,
        "resume_run_dir": str(args.resume_run_dir) if getattr(args, "resume_run_dir", None) else "",
    }
    store.write_config(config_payload)
    store.write_run_meta(
        {
            "run_id": run_id,
            "created_at": config_payload["created_at"],
            "git_commit": config_payload["git_commit"],
            "git_dirty": config_payload["git_dirty"],
            "scorer_version": SCORER_VERSION,
            "result_schema_version": RESULT_ROW_SCHEMA_VERSION,
            "summary_schema_version": SUMMARY_ROW_SCHEMA_VERSION,
            **runtime_metadata(),
            "workflow_sha256": workflow_sha,
            "task_yaml_sha256": task_yaml_sha,
            "prompt_sha256": prompt_sha,
        }
    )
    store.write_dataset_fingerprint(dataset_fingerprint(args.dataset))
    store.write_model_snapshot({"models": model_snapshot, "model_registry": str(args.model_registry)})

    if args.task:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
        if not model_ids:
            model_ids = [registry.active] if registry.active else []
        if not model_ids:
            raise ValueError("No model ids supplied and no active model in registry")
        if completed_keys:
            repeats = max(1, int(args.repeats))
            filtered_dataset: list[dict[str, object]] = []
            for sample in dataset:
                sample_id = str(sample.get("sample_id", ""))
                all_done = True
                for model_id in model_ids:
                    for repeat_idx in range(repeats):
                        key = (sample_id, args.task, STEP_ID_TASK, model_id, repeat_idx)
                        if key not in completed_keys:
                            all_done = False
                            break
                    if not all_done:
                        break
                if not all_done:
                    filtered_dataset.append(sample)
            dataset = filtered_dataset
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
            completed_keys=completed_keys,
        )
    else:
        assert workflow_spec is not None
        workflow = workflow_spec
        default_model_id, step_model_map = parse_workflow_models(args.models, registry)
        if completed_keys:
            already_done_samples = {
                key[0]
                for key in completed_keys
                if key[1] == workflow.name and key[2] == STEP_ID_E2E
            }
            dataset = [s for s in dataset if str(s.get("sample_id", "")) not in already_done_samples]
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
            completed_keys=completed_keys,
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
    parser.add_argument("--resume-run-dir", default=None, help="Resume and append to an existing run dir")
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
