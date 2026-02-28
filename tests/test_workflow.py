import json

import pytest

from eval.workflow import load_workflow


def test_load_workflow(tmp_path) -> None:
    path = tmp_path / "workflow.json"
    path.write_text(
        json.dumps(
            {
                "name": "news_pipeline",
                "version": "v1",
                "steps": [
                    {"id": "step1_dedup", "task": "news_dedup", "model_id": "m1"},
                    {"id": "step2_extract", "task": "ie_json", "model_id": "m2"},
                    {"id": "step3_score", "task": "stock_score", "model_id": "m3"},
                ],
            }
        ),
        encoding="utf-8",
    )
    wf = load_workflow(path)
    assert wf.name == "news_pipeline"
    assert len(wf.steps) == 3
    assert wf.steps[0].id == "step1_dedup"


def test_load_workflow_rejects_non_object_root(tmp_path) -> None:
    path = tmp_path / "bad_workflow.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Workflow root must be an object"):
        load_workflow(path)


def test_load_workflow_rejects_duplicate_step_id(tmp_path) -> None:
    path = tmp_path / "dup_step.json"
    path.write_text(
        json.dumps(
            {
                "name": "wf_dup",
                "steps": [
                    {"id": "s1", "task": "news_dedup"},
                    {"id": "s1", "task": "ie_json", "input_from": "s1"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate step id"):
        load_workflow(path)


def test_load_workflow_rejects_unknown_input_from(tmp_path) -> None:
    path = tmp_path / "bad_ref.json"
    path.write_text(
        json.dumps(
            {
                "name": "wf_bad_ref",
                "steps": [
                    {"id": "s1", "task": "news_dedup"},
                    {"id": "s2", "task": "ie_json", "input_from": "not_exists"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown input_from"):
        load_workflow(path)


def test_load_workflow_rejects_cycle_dependency(tmp_path) -> None:
    path = tmp_path / "cycle_ref.json"
    path.write_text(
        json.dumps(
            {
                "name": "wf_cycle",
                "steps": [
                    {"id": "s1", "task": "news_dedup", "input_from": "s2"},
                    {"id": "s2", "task": "ie_json", "input_from": "s1"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cycle"):
        load_workflow(path)
