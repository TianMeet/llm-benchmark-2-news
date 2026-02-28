"""GenericTask 指标扩展测试。"""

from __future__ import annotations

from pathlib import Path

from eval.tasks.generic import GenericTask


def _write_task_yaml(tmp_path: Path) -> Path:
    cfg = """
name: generic_metric_test
version: v1
prompt_template: ie_json

parse_schema:
  - field: sentiment
    type: enum
    values: [positive, neutral, negative]
    default: neutral
  - field: impact_score
    type: int
    default: 0
    lo: -5
    hi: 5
  - field: keywords
    type: list
    default: []

metrics:
  - type: exact_match
    name: sentiment_acc
    pred_field: sentiment
    label_field: gt_sentiment
  - type: numeric_error
    name: impact
    pred_field: impact_score
    label_field: gt_impact_score
    tolerance: 1
  - type: list_overlap
    name: keyword
    pred_field: keywords
    label_field: gt_keywords
"""
    path = tmp_path / "task.yaml"
    path.write_text(cfg.strip() + "\n", encoding="utf-8")
    return path


def _write_reasoning_task_yaml(tmp_path: Path) -> Path:
    cfg = """
name: reasoning_metric_test
version: v1
prompt_template: ie_json

parse_schema:
  - field: final_answer
    type: str
    default: ""
  - field: steps
    type: list
    default: []
  - field: claims
    type: list_raw
    default: []

metrics:
  - type: reasoning_consistency
    name: reasoning
    final_answer_field: final_answer
    support_fields: [steps, claims]
  - type: claim_evidence_overlap
    name: evidence
    claims_field: claims
    label_field: gt_evidence_ids
    claim_evidence_key: evidence
"""
    path = tmp_path / "reasoning_task.yaml"
    path.write_text(cfg.strip() + "\n", encoding="utf-8")
    return path


def test_generic_task_extended_metrics(tmp_path: Path) -> None:
    task = GenericTask(_write_task_yaml(tmp_path))
    sample = {
        "gt_sentiment": "Positive",
        "gt_impact_score": 2,
        "gt_keywords": ["AI", "Earnings", "Guidance"],
    }
    parsed = task.parse('{"sentiment":"positive","impact_score":3,"keywords":["ai","earnings","macro"]}')
    assert parsed is not None

    m = task.metrics(sample, "", parsed)
    assert m["sentiment_acc"] == 1.0
    assert m["impact_mae"] == 1.0
    assert m["impact_within_tol"] == 1.0
    assert m["keyword_precision"] == 0.6667
    assert m["keyword_recall"] == 0.6667
    assert m["keyword_f1"] == 0.6667


def test_numeric_error_handles_missing_label(tmp_path: Path) -> None:
    task = GenericTask(_write_task_yaml(tmp_path))
    parsed = task.parse('{"sentiment":"neutral","impact_score":1,"keywords":[]}')
    assert parsed is not None

    m = task.metrics({"gt_sentiment": "neutral", "gt_keywords": []}, "", parsed)
    assert m["impact_mae"] is None
    assert m["impact_within_tol"] == 0.0


def test_reasoning_consistency_and_claim_evidence_overlap(tmp_path: Path) -> None:
    task = GenericTask(_write_reasoning_task_yaml(tmp_path))
    sample = {"gt_evidence_ids": ["n1", "n3", "n4"]}
    parsed = task.parse(
        '{"final_answer":"positive","steps":["Given earnings beat, sentiment is positive."],'
        '"claims":[{"text":"Revenue increased","evidence":["n1","n2"]},{"text":"Margin up","evidence":["n3"]}]}'
    )
    assert parsed is not None

    m = task.metrics(sample, "", parsed)
    assert m["reasoning_consistency"] == 1.0
    assert m["reasoning_support_coverage"] == 0.3333
    assert m["evidence_precision"] == 0.6667
    assert m["evidence_recall"] == 0.6667
    assert m["evidence_f1"] == 0.6667


def test_reasoning_consistency_miss(tmp_path: Path) -> None:
    task = GenericTask(_write_reasoning_task_yaml(tmp_path))
    parsed = task.parse(
        '{"final_answer":"negative","steps":["Earnings are strong and outlook is positive."],"claims":[]}'
    )
    assert parsed is not None

    m = task.metrics({"gt_evidence_ids": []}, "", parsed)
    assert m["reasoning_consistency"] == 0.0
    assert m["reasoning_support_coverage"] == 0.0
