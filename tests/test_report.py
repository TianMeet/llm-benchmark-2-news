import csv
import json

from eval.report import generate_report


def test_generate_report(tmp_path) -> None:
    run_dir = tmp_path / "run1"
    run_dir.mkdir()

    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
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
        )
        w.writeheader()
        w.writerow(
            {
                "task_name": "ie_json",
                "step_id": "__task__",
                "model_id": "m1",
                "request_count": "1",
                "success_rate": "1.0",
                "parse_rate": "1.0",
                "avg_latency_ms": "100",
                "p95_latency_ms": "100",
                "avg_total_tokens": "42",
                "cost_estimate": "0.0",
            }
        )

    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"sample_id": "s1", "success": False, "parse_success": False, "task_name": "ie_json", "step_id": "__task__", "model_id": "m1", "error": "x", "output_text": "bad"}) + "\n")

    report_path = generate_report(run_dir)
    text = report_path.read_text(encoding="utf-8")
    assert "LLM Eval Report" in text
    assert "Case 1" in text
