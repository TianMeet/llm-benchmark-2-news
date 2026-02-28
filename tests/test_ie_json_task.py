"""ie_json 任务测试 — 通过 build_task() 加载 YAML 驱动的 GenericTask。"""

from bench.tasks import build_task


def test_ie_json_parse_and_metrics() -> None:
    task = build_task("ie_json")
    output = '{"ticker":"AAPL","event_type":"earnings","sentiment":"positive","impact_score":4,"summary":"业绩超预期"}'
    parsed = task.parse(output)
    assert parsed is not None
    assert parsed["ticker"] == "AAPL"
    assert parsed["impact_score"] == 4

    metrics = task.metrics({"ticker": "AAPL"}, output, parsed)
    assert metrics["field_completeness"] == 1.0


def test_new_task_auto_discovered() -> None:
    """bench/tasks/ 下任意 YAML 文件均可被自动发现，无需改 Python 代码。"""
    task = build_task("stock_score")
    assert task.name == "stock_score"
    assert task.parse_applicable is True


def test_unknown_task_raises() -> None:
    import pytest
    with pytest.raises(ValueError, match="Unknown task"):
        build_task("nonexistent_task_xyz")
