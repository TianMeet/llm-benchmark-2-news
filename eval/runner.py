"""CLI runner for single-task and workflow-based model evaluation."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.cache import EvalCache
from eval.metrics import aggregate_records
from eval.registry import ModelRegistry, estimate_cost
from eval.report import generate_report
from eval.tasks import build_task
from eval.workflow import WorkflowSpec, load_workflow


@dataclass
class UnifiedCallResult:
    content: str
    usage: dict[str, int]
    latency_ms: float
    raw_response: dict[str, Any]
    success: bool
    error: str
    retry_count: int
    from_cache: bool = False


def _load_dataset(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    if max_samples == 0:
        return []
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def _git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out
    except Exception:
        return "unknown"


async def _call_model(
    registry: ModelRegistry,
    model_id: str,
    messages: list[dict[str, Any]],
    params_override: dict[str, Any],
    cache: EvalCache | None,
    sample_id: str,
) -> UnifiedCallResult:
    cache_key = ""
    if cache is not None:
        cache_key = cache.make_key(
            {
                "model_id": model_id,
                "params": params_override,
                "messages": messages,
                "sample_id": sample_id,
            }
        )
        cached = cache.get(cache_key)
        if cached:
            return UnifiedCallResult(**cached, from_cache=True)

    client = registry.create_client(model_id, params_override=params_override)
    llm_resp = await client.complete(messages)
    usage = {
        "prompt_tokens": int(llm_resp.input_tokens),
        "completion_tokens": int(llm_resp.output_tokens),
        "total_tokens": int(llm_resp.input_tokens + llm_resp.output_tokens),
    }
    result = UnifiedCallResult(
        content=llm_resp.text,
        usage=usage,
        latency_ms=float(llm_resp.latency_ms),
        raw_response={
            "model": llm_resp.model,
            "success": llm_resp.success,
            "error_message": llm_resp.error_message,
        },
        success=bool(llm_resp.success),
        error=str(llm_resp.error_message or ""),
        retry_count=int(getattr(llm_resp, "retry_count", 0)),
    )
    if cache is not None and cache_key:
        cache.set(cache_key, asdict(result))
    return result


def _mock_response(task_name: str, sample: dict[str, Any]) -> UnifiedCallResult:
    ticker = str(sample.get("ticker", "UNKNOWN"))
    if task_name == "ie_json":
        content = json.dumps(
            {
                "ticker": ticker,
                "event_type": "other",
                "sentiment": "neutral",
                "impact_score": 0,
                "summary": "mock",
            },
            ensure_ascii=False,
        )
    elif task_name == "stock_score":
        content = json.dumps({"score": 0, "rationale": "mock"}, ensure_ascii=False)
    elif task_name == "news_dedup":
        titles = [str(n.get("title", "")) for n in sample.get("news_items", [])]
        deduped = list(dict.fromkeys(titles))
        content = json.dumps(
            {"deduped_titles": deduped, "removed_count": max(0, len(titles) - len(deduped))},
            ensure_ascii=False,
        )
    else:
        content = "{}"
    return UnifiedCallResult(
        content=content,
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        latency_ms=1.0,
        raw_response={"model": "mock", "success": True, "error_message": ""},
        success=True,
        error="",
        retry_count=0,
        from_cache=False,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


async def _run_task_mode(
    run_id: str,
    run_dir: Path,
    dataset: list[dict[str, Any]],
    registry: ModelRegistry,
    task_name: str,
    model_ids: list[str],
    params_override: dict[str, Any],
    repeats: int,
    cache: EvalCache | None,
    mock: bool,
) -> list[dict[str, Any]]:
    task = build_task(task_name)
    results_path = run_dir / "results.jsonl"
    records: list[dict[str, Any]] = []

    for model_id in model_ids:
        spec = registry.get(model_id)
        for sample in dataset:
            sample_id = str(sample.get("sample_id", ""))
            for repeat_idx in range(repeats):
                if mock:
                    messages, prompt_version = [], "mock"
                    call = _mock_response(task.name, sample)
                else:
                    messages, prompt_version = task.build_prompt(sample, context={"repeat_idx": repeat_idx})
                    call = await _call_model(
                        registry=registry,
                        model_id=model_id,
                        messages=messages,
                        params_override=params_override,
                        cache=cache,
                        sample_id=f"{sample_id}:{repeat_idx}:{task.name}",
                    )
                parsed = task.parse(call.content) if call.success else None
                task_metrics = task.metrics(sample, call.content, parsed)
                parse_success = parsed is not None
                cost = estimate_cost(call.usage, spec.price_config)
                row = {
                    "run_id": run_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "mode": "task",
                    "sample_id": sample_id,
                    "task_name": task.name,
                    "task_version": task.version,
                    "step_id": "__task__",
                    "prompt_version": prompt_version,
                    "model_id": model_id,
                    "params": params_override,
                    "repeat_idx": repeat_idx,
                    "success": call.success,
                    "error": call.error,
                    "parse_applicable": task.parse_applicable,
                    "parse_success": parse_success,
                    "latency_ms": round(call.latency_ms, 2),
                    "prompt_tokens": call.usage["prompt_tokens"],
                    "completion_tokens": call.usage["completion_tokens"],
                    "total_tokens": call.usage["total_tokens"],
                    "retry_count": call.retry_count,
                    "cost_estimate": round(cost, 8),
                    "from_cache": call.from_cache,
                    "output_text": call.content,
                    "parsed": parsed,
                    "task_metrics": task_metrics,
                }
                _append_jsonl(results_path, row)
                records.append(row)

    return records


async def _run_workflow_mode(
    run_id: str,
    run_dir: Path,
    dataset: list[dict[str, Any]],
    registry: ModelRegistry,
    workflow: WorkflowSpec,
    global_params: dict[str, Any],
    cache: EvalCache | None,
    mock: bool,
) -> list[dict[str, Any]]:
    results_path = run_dir / "results.jsonl"
    records: list[dict[str, Any]] = []

    for sample in dataset:
        sample_id = str(sample.get("sample_id", ""))
        step_outputs: dict[str, dict[str, Any]] = {}
        e2e_success = True
        e2e_latency = 0.0
        e2e_tokens = 0
        e2e_retry = 0
        final_output: dict[str, Any] | None = None

        for step in workflow.steps:
            task = build_task(step.task)
            if step.input_from == "sample":
                step_input = sample
            else:
                step_input = step_outputs.get(step.input_from, {}).get("parsed") or sample

            if mock:
                messages, prompt_version = [], "mock"
            else:
                messages, prompt_version = task.build_prompt(sample, context={"input": step_input, "step_outputs": step_outputs})
            merged_params = dict(global_params)
            merged_params.update(step.params_override)

            if mock:
                call = _mock_response(task.name, sample)
            else:
                call = await _call_model(
                    registry=registry,
                    model_id=step.model_id,
                    messages=messages,
                    params_override=merged_params,
                    cache=cache,
                    sample_id=f"{sample_id}:{step.id}:{task.name}",
                )
            parsed = task.parse(call.content) if call.success else None
            parse_success = parsed is not None
            task_metrics = task.metrics(sample, call.content, parsed)
            spec = registry.get(step.model_id)
            cost = estimate_cost(call.usage, spec.price_config)

            e2e_success = e2e_success and call.success and (parse_success or not task.parse_applicable)
            e2e_latency += call.latency_ms
            e2e_tokens += call.usage["total_tokens"]
            e2e_retry += call.retry_count
            if parsed is not None:
                final_output = parsed

            step_row = {
                "run_id": run_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "mode": "workflow_step",
                "workflow_name": workflow.name,
                "workflow_version": workflow.version,
                "sample_id": sample_id,
                "task_name": task.name,
                "task_version": task.version,
                "step_id": step.id,
                "prompt_version": prompt_version,
                "model_id": step.model_id,
                "params": merged_params,
                "success": call.success,
                "error": call.error,
                "parse_applicable": task.parse_applicable,
                "parse_success": parse_success,
                "latency_ms": round(call.latency_ms, 2),
                "prompt_tokens": call.usage["prompt_tokens"],
                "completion_tokens": call.usage["completion_tokens"],
                "total_tokens": call.usage["total_tokens"],
                "retry_count": call.retry_count,
                "cost_estimate": round(cost, 8),
                "from_cache": call.from_cache,
                "output_text": call.content,
                "parsed": parsed,
                "task_metrics": task_metrics,
            }
            _append_jsonl(results_path, step_row)
            records.append(step_row)

            step_outputs[step.id] = {
                "parsed": parsed,
                "raw": call.content,
                "success": call.success,
                "parse_success": parse_success,
            }

        e2e_row = {
            "run_id": run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "mode": "workflow_e2e",
            "workflow_name": workflow.name,
            "workflow_version": workflow.version,
            "sample_id": sample_id,
            "task_name": workflow.name,
            "task_version": workflow.version,
            "step_id": "__e2e__",
            "prompt_version": "workflow",
            "model_id": "+".join(step.model_id for step in workflow.steps),
            "params": global_params,
            "success": e2e_success,
            "error": "" if e2e_success else "workflow_step_failed",
            "parse_applicable": True,
            "parse_success": final_output is not None,
            "latency_ms": round(e2e_latency, 2),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": e2e_tokens,
            "retry_count": e2e_retry,
            "cost_estimate": 0.0,
            "from_cache": False,
            "output_text": "",
            "parsed": final_output,
            "task_metrics": {},
        }
        _append_jsonl(results_path, e2e_row)
        records.append(e2e_row)

    return records


def _parse_params(params_str: str | None) -> dict[str, Any]:
    if not params_str:
        return {}
    return dict(json.loads(params_str))


def _build_run_dir(base_out: str | Path) -> tuple[str, Path]:
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(base_out) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


async def run(args: argparse.Namespace) -> Path:
    if bool(args.task) == bool(args.workflow):
        raise ValueError("Exactly one of --task or --workflow must be provided")

    run_id, run_dir = _build_run_dir(args.out)
    dataset = _load_dataset(args.dataset, max_samples=args.max_samples)
    registry = ModelRegistry(args.model_registry)
    global_params = _parse_params(args.params)
    cache = None if args.no_cache else EvalCache(args.cache_dir)

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
        "git_commit": _git_commit_hash(),
        "env_keys": sorted([k for k in os.environ.keys() if k.endswith("API_KEY")]),
    }
    _write_json(run_dir / "config.json", config_payload)

    if args.task:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
        if not model_ids:
            model_ids = [registry.active] if registry.active else []
        if not model_ids:
            raise ValueError("No model ids supplied and no active model in registry")
        records = await _run_task_mode(
            run_id=run_id,
            run_dir=run_dir,
            dataset=dataset,
            registry=registry,
            task_name=args.task,
            model_ids=model_ids,
            params_override=global_params,
            repeats=max(1, int(args.repeats)),
            cache=cache,
            mock=args.mock,
        )
    else:
        workflow = load_workflow(args.workflow)
        records = await _run_workflow_mode(
            run_id=run_id,
            run_dir=run_dir,
            dataset=dataset,
            registry=registry,
            workflow=workflow,
            global_params=global_params,
            cache=cache,
            mock=args.mock,
        )

    summary_rows = aggregate_records(records)
    _write_summary_csv(run_dir / "summary.csv", summary_rows)
    generate_report(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM benchmark runner")
    parser.add_argument("--task", help="Task name, e.g. ie_json")
    parser.add_argument("--workflow", help="Workflow YAML/JSON path")
    parser.add_argument("--dataset", required=True, help="Dataset JSONL path")
    parser.add_argument("--models", default="", help="Comma-separated model IDs")
    parser.add_argument("--model-registry", default="configs/llm_providers.yaml")
    parser.add_argument("--out", default="runs/")
    parser.add_argument("--params", default=None, help='JSON override, e.g. {"temperature":0.2}')
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cache-dir", default=".eval_cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock responses without LLM API calls")
    args = parser.parse_args()

    run_dir = asyncio.run(run(args))
    print(str(run_dir))


if __name__ == "__main__":
    main()
