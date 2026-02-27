"""Report generation from run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join(["---"] * len(fields)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    return "\n".join([header, sep, *body])


def generate_report(run_dir: str | Path) -> Path:
    run_path = Path(run_dir)
    summary_rows = _read_csv(run_path / "summary.csv")
    records = _read_jsonl(run_path / "results.jsonl")

    errors = [
        r
        for r in records
        if (not r.get("success", False)) or (r.get("parse_applicable", True) and not r.get("parse_success", False))
    ]

    lines = [
        "# LLM Eval Report",
        "",
        f"Run Dir: `{run_path}`",
        "",
        "## Model x Metric Summary",
        "",
        _markdown_table(
            summary_rows,
            [
                "task_name",
                "step_id",
                "model_id",
                "request_count",
                "success_rate",
                "parse_rate",
                "avg_latency_ms",
                "p95_latency_ms",
                "avg_total_tokens",
                "cost_estimate",
            ],
        ),
        "",
        "## Top Error Cases (max 3)",
        "",
    ]

    for idx, err in enumerate(errors[:3], start=1):
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
    parser = argparse.ArgumentParser(description="Generate markdown report from run_dir")
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    out = generate_report(args.run_dir)
    print(str(out))


if __name__ == "__main__":
    main()
