from eval.tasks.ie_json import IEJsonTask


def test_ie_json_parse_and_metrics() -> None:
    task = IEJsonTask()
    output = '{"ticker":"AAPL","event_type":"earnings","sentiment":"positive","impact_score":4,"summary":"业绩超预期"}'
    parsed = task.parse(output)
    assert parsed is not None
    assert parsed["ticker"] == "AAPL"
    assert parsed["impact_score"] == 4

    metrics = task.metrics({"ticker": "AAPL"}, output, parsed)
    assert metrics["field_completeness"] == 1.0
