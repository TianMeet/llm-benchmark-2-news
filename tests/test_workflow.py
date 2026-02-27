import json

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
