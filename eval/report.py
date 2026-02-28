"""从运行产物生成 Markdown 报告。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件为字典列表。"""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """读取 CSV 文件为字典列表。"""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    """将结构化数据渲染为 Markdown 表格。"""
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join(["---"] * len(fields)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    return "\n".join([header, sep, *body])


def generate_report(run_dir: str | Path) -> Path:
    """生成 report.md，包含指标对比与失败样本摘要。"""
    run_path = Path(run_dir)
    summary_rows = _read_csv(run_path / "summary.csv")
    records = _read_jsonl(run_path / "results.jsonl")

    errors = [
        r
        for r in records
        if (not r.get("success", False)) or (r.get("parse_applicable", True) and not r.get("parse_success", False))
    ]

    # 按 task 分组 summary 行，用于模型横向对比
    task_summaries: dict[str, list[dict]] = defaultdict(list)
    for row in summary_rows:
        task_summaries[str(row.get("task_name", ""))].append(row)

    # 收集各 task 下的 tm_* 指标列名
    tm_fields_by_task: dict[str, list[str]] = {
        task: sorted({k for row in rows for k in row if k.startswith("tm_")})
        for task, rows in task_summaries.items()
    }

    lines: list[str] = [
        "# LLM Eval Report",
        "",
        f"Run Dir: `{run_path}`",
        "",
        "## 总表：Model x Metric",
        "",
        _markdown_table(
            summary_rows,
            [
                "task_name",
                "model_id",
                "request_count",
                "success_rate",
                "parse_rate",
                "avg_latency_ms",
                "avg_total_tokens",
                "cost_estimate",
            ],
        ),
        "",
    ]

    # 每个 task 单独输出模型对比表（含 tm_* 任务指标）
    if task_summaries:
        lines.append("## 按任务维度模型对比")
        lines.append("")
        for task_name, rows in sorted(task_summaries.items()):
            tm_fields = tm_fields_by_task.get(task_name, [])
            compare_fields = (
                ["model_id", "success_rate", "parse_rate",
                 "avg_latency_ms", "cost_estimate"]
                + tm_fields
            )
            lines.append(f"### {task_name}")
            lines.append("")
            lines.append(_markdown_table(rows, compare_fields))
            lines.append("")
            # 指向细粒度分区文件
            by_model_dir = run_path / "by_model" / task_name
            if by_model_dir.exists():
                model_files = sorted(by_model_dir.glob("*.jsonl"))
                if model_files:
                    lines.append("细粒度结果文件：")
                    for mf in model_files:
                        lines.append(f"- `{mf.relative_to(run_path)}`")
                    lines.append("")

    workflow_rows = [r for r in records if str(r.get("mode", "")).startswith("workflow_")]
    if workflow_rows:
        grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in workflow_rows:
            key = (
                str(r.get("workflow_name", "")),
                str(r.get("workflow_version", "")),
                str(r.get("mode", "")),
                str(r.get("model_id", "")),
            )
            grouped[key].append(r)

        wf_summary: list[dict[str, Any]] = []
        for (wf_name, wf_ver, mode, model_id), items in sorted(grouped.items()):
            request_count = len(items)
            success_count = sum(1 for i in items if i.get("success"))
            parse_items = [i for i in items if i.get("parse_applicable", True)]
            parse_success_count = sum(1 for i in parse_items if i.get("parse_success"))
            avg_latency = (
                sum(float(i.get("latency_ms", 0.0)) for i in items) / request_count
                if request_count
                else 0.0
            )
            total_cost = sum(float(i.get("cost_estimate", 0.0)) for i in items)
            wf_summary.append(
                {
                    "workflow_name": wf_name,
                    "workflow_version": wf_ver,
                    "mode": mode,
                    "model_id": model_id,
                    "request_count": request_count,
                    "success_rate": round(success_count / request_count, 4) if request_count else 0.0,
                    "parse_rate": (
                        round(parse_success_count / len(parse_items), 4) if parse_items else 0.0
                    ),
                    "avg_latency_ms": round(avg_latency, 2),
                    "cost_estimate": round(total_cost, 6),
                }
            )

        lines.append("## 按 Workflow 维度概览")
        lines.append("")
        lines.append(
            _markdown_table(
                wf_summary,
                [
                    "workflow_name",
                    "workflow_version",
                    "mode",
                    "model_id",
                    "request_count",
                    "success_rate",
                    "parse_rate",
                    "avg_latency_ms",
                    "cost_estimate",
                ],
            )
        )
        lines.append("")

    lines += [
        "## 失败样本 (max 5)",
        "",
    ]

    for idx, err in enumerate(errors[:5], start=1):
        lines.extend(
            [
                f"### Case {idx}",
                f"- sample_id: `{err.get('sample_id', '')}`",
                f"- task/step: `{err.get('task_name', '')}` / `{err.get('step_id', '')}`",
                f"- model: `{err.get('model_id', '')}`",
                f"- error: `{err.get('error', '')}`",
                f"- output_excerpt: `{str(err.get('output_text', ''))[:180]}`",
                "",
            ]
        )

    if not errors:
        lines.append("No error cases found.")

    report_path = run_path / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    """命令行入口：按 run_dir 生成报告。"""
    parser = argparse.ArgumentParser(description="Generate markdown report from run_dir")
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    out = generate_report(args.run_dir)
    print(str(out))


if __name__ == "__main__":
    main()
