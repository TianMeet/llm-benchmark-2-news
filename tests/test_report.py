import csv
import json

from bench.reporting.report import generate_report


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


def test_generate_report_handles_missing_input_files(tmp_path) -> None:
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    report_path = generate_report(run_dir)
    text = report_path.read_text(encoding="utf-8")
    assert "LLM Eval Report" in text
    assert "No error cases found." in text


def test_generate_report_includes_workflow_grouping(tmp_path) -> None:
    run_dir = tmp_path / "workflow_run"
    run_dir.mkdir()

    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "schema_version",
                "task_name",
                "step_id",
                "model_id",
                "request_count",
                "success_rate",
                "parse_rate",
                "avg_latency_ms",
                "avg_total_tokens",
                "cost_estimate",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "schema_version": "summary_row.v1",
                "task_name": "news_pipeline",
                "step_id": "__e2e__",
                "model_id": "m1+m2",
                "request_count": "1",
                "success_rate": "1.0",
                "parse_rate": "1.0",
                "avg_latency_ms": "50",
                "avg_total_tokens": "100",
                "cost_estimate": "0.01",
            }
        )

    rows = [
        {
            "schema_version": "result_row.v1",
            "mode": "workflow_step",
            "workflow_name": "news_pipeline",
            "workflow_version": "v2",
            "task_name": "news_dedup",
            "step_id": "step1",
            "model_id": "m1",
            "success": True,
            "parse_applicable": True,
            "parse_success": True,
            "latency_ms": 20,
            "cost_estimate": 0.001,
        },
        {
            "schema_version": "result_row.v1",
            "mode": "workflow_e2e",
            "workflow_name": "news_pipeline",
            "workflow_version": "v2",
            "task_name": "news_pipeline",
            "step_id": "__e2e__",
            "model_id": "m1+m2",
            "success": True,
            "parse_applicable": True,
            "parse_success": True,
            "latency_ms": 50,
            "cost_estimate": 0.01,
        },
    ]
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_path = generate_report(run_dir)
    text = report_path.read_text(encoding="utf-8")
    assert "按 Workflow 维度概览" in text
    assert "news_pipeline" in text
    assert "v2" in text
