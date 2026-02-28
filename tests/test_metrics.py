from eval.metrics.aggregate import aggregate_records


def test_aggregate_records_basic() -> None:
    records = [
        {
            "task_name": "ie_json",
            "model_id": "m1",
            "step_id": "__task__",
            "success": True,
            "parse_applicable": True,
            "parse_success": True,
            "latency_ms": 100,
            "total_tokens": 40,
            "retry_count": 1,
            "cost_estimate": 0.01,
        },
        {
            "task_name": "ie_json",
            "model_id": "m1",
            "step_id": "__task__",
            "success": False,
            "parse_applicable": True,
            "parse_success": False,
            "latency_ms": 200,
            "total_tokens": 60,
            "retry_count": 2,
            "cost_estimate": 0.02,
        },
    ]
    rows = aggregate_records(records)
    assert len(rows) == 1
    row = rows[0]
    assert row["request_count"] == 2
    assert row["success_count"] == 1
    assert row["success_rate"] == 0.5
    assert row["parse_rate"] == 0.5
    assert row["retry_count"] == 3
    assert row["schema_version"] == "summary_row.v1"
