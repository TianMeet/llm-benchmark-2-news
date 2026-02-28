"""评测执行 CLI 入口——支持单任务评测与多步 Workflow 评测。

使用方式：
  # 单任务
  python -m bench.cli.runner --task news_analysis --dataset ... --models deepseek-chat --out runs/
  # Workflow
  python -m bench.cli.runner --workflow workflows/news_pipeline.yaml --dataset ... --out runs/

执行流程：
  1. 参数解析与校验
  2. RunStore 初始化（runs/<timestamp>/ 目录）
  3. 数据集加载 + 模型注册表解析
  4. 写入 config / run_meta / dataset_fingerprint / model_snapshot
  5. 调用 run_task_mode() 或 run_workflow_mode() 执行评测
  6. aggregate_records() 聚合指标
  7. 写入 summary.csv + 分区结果 + report.md

支持功能：
- --mock          离线模式（不触发真实 API）
- --no-cache      禁用缓存
- --resume-run-dir 断点续跑
- --concurrency   task 模式并发控制
- --workflow-concurrency  workflow 样本级并发
"""

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


def _resolve_model_ids(args: argparse.Namespace, registry: ModelRegistry) -> list[str]:
    """从 args.models 解析 model_id 列表，回退到 registry.active。"""
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_ids:
        model_ids = [registry.active] if registry.active else []
    return model_ids


def _collect_task_snapshot(
    args: argparse.Namespace,
    registry: ModelRegistry,
    model_ids: list[str],
    global_params: dict[str, object],
) -> tuple[dict[str, dict[str, object]], str, str]:
    """构建 task 模式的 model_snapshot 与 SHA 指纹。返回 (snapshot, task_yaml_sha, prompt_sha)。"""
    model_snapshot: dict[str, dict[str, object]] = {}
    task_yaml_sha = ""
    prompt_sha = ""
    task_yaml = Path(__file__).resolve().parents[1] / "tasks" / f"{args.task}.yaml"
    if task_yaml.exists():
        task_yaml_sha = sha256_file(task_yaml)
        try:
            import yaml
            raw_task_cfg = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
            prompt_name = str(raw_task_cfg.get("prompt_template", ""))
            if prompt_name:
                prompt_path = Path(__file__).resolve().parents[1] / "prompts" / f"{prompt_name}.yaml"
                if prompt_path.exists():
                    prompt_sha = sha256_file(prompt_path)
        except Exception:
            pass
    for mid in model_ids:
        resolved = dict(registry.resolve_config(mid, global_params))
        model_snapshot[mid] = redact_sensitive_fields(resolved)
    return model_snapshot, task_yaml_sha, prompt_sha


def _collect_workflow_snapshot(
    args: argparse.Namespace,
    registry: ModelRegistry,
    workflow_spec: object,
    global_params: dict[str, object],
) -> tuple[dict[str, dict[str, object]], str]:
    """构建 workflow 模式的 model_snapshot 与 workflow SHA。返回 (snapshot, workflow_sha)。"""
    from bench.workflow import WorkflowSpec

    assert isinstance(workflow_spec, WorkflowSpec)
    model_snapshot: dict[str, dict[str, object]] = {}
    workflow_sha = sha256_file(args.workflow)
    default_model_id, step_model_map = parse_workflow_models(args.models, registry)
    for step in workflow_spec.steps:
        model_id = step_model_map.get(step.id) or (
            default_model_id if step.model_id == "$active" else step.model_id
        )
        if not model_id:
            continue
        merged = dict(registry.resolve_config(model_id, {**global_params, **step.params_override}))
        model_snapshot[f"{step.id}:{model_id}"] = redact_sensitive_fields(merged)
    return model_snapshot, workflow_sha


def _build_config_payload(
    run_id: str,
    args: argparse.Namespace,
    global_params: dict[str, object],
) -> dict[str, object]:
    """构建本次运行的 config.json 载荷。"""
    return {
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


def _filter_completed_task_samples(
    dataset: list[dict[str, object]],
    task_name: str,
    model_ids: list[str],
    repeats: int,
    completed_keys: set[tuple[str, str, str, str, int]],
) -> list[dict[str, object]]:
    """过滤已完成的样本，用于断点续跑。"""
    if not completed_keys:
        return dataset
    filtered: list[dict[str, object]] = []
    for sample in dataset:
        sample_id = str(sample.get("sample_id", ""))
        all_done = True
        for model_id in model_ids:
            for repeat_idx in range(repeats):
                key = (sample_id, task_name, STEP_ID_TASK, model_id, repeat_idx)
                if key not in completed_keys:
                    all_done = False
                    break
            if not all_done:
                break
        if not all_done:
            filtered.append(sample)
    return filtered


async def run(args: argparse.Namespace) -> Path:
    """主执行函数：完成评测、聚合与报告生成。"""
    if bool(args.task) == bool(args.workflow):
        raise ValueError("Exactly one of --task or --workflow must be provided")

    if getattr(args, "resume_run_dir", None):
        store = RunStore(args.out, existing_run_dir=args.resume_run_dir)
    else:
        store = RunStore(args.out)
    # ── 为本次 run 添加文件日志 handler，便于事后排查 ──────────────
    from bench_utils.logging_config import setup_logging
    setup_logging(log_file=store.run_dir / "run.log")

    run_id = store.run_id
    dataset = load_dataset(args.dataset, max_samples=args.max_samples)
    registry = ModelRegistry(args.model_registry)
    global_params = parse_params(args.params)
    cache = None if args.no_cache else EvalCache(args.cache_dir)
    gateway = LLMGateway(registry=registry, cache=cache, mock=args.mock)
    completed_keys = store.load_completed_keys() if getattr(args, "resume_run_dir", None) else set()
    workflow_spec = load_workflow(args.workflow) if args.workflow else None

    # ── 构建 model_snapshot 与 SHA 指纹 ──────────────────────────
    workflow_sha = ""
    task_yaml_sha = ""
    prompt_sha = ""

    if args.task:
        model_ids = _resolve_model_ids(args, registry)
        model_snapshot, task_yaml_sha, prompt_sha = _collect_task_snapshot(
            args, registry, model_ids, global_params,
        )
    else:
        model_snapshot, workflow_sha = _collect_workflow_snapshot(
            args, registry, workflow_spec, global_params,
        )

    # ── 写入审计产物 ─────────────────────────────────────────────
    config_payload = _build_config_payload(run_id, args, global_params)
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

    # ── 执行评测 ────────────────────────────────────────────────
    if args.task:
        model_ids = _resolve_model_ids(args, registry)
        if not model_ids:
            raise ValueError("No model ids supplied and no active model in registry")
        repeats = max(1, int(args.repeats))
        dataset = _filter_completed_task_samples(
            dataset, args.task, model_ids, repeats, completed_keys,
        )
        records = await run_task_mode(
            run_id=run_id,
            store=store,
            dataset=dataset,
            registry=registry,
            task_name=args.task,
            model_ids=model_ids,
            params_override=global_params,
            repeats=repeats,
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
    store.close()
    MarkdownReporter().generate(store.run_dir)
    logger.info(
        "Run completed: %s  records=%d  summary_rows=%d",
        store.run_dir, len(records), len(summary_rows),
    )
    return store.run_dir


def main() -> None:
    """CLI 入口。"""
    # ── 日志初始化（必须在 argparse 之前，这样参数解析错误也有日志格式） ──
    from bench_utils.logging_config import setup_logging
    setup_logging()

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
    parser.add_argument("--model-registry", default="configs/llm_providers.json")
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
