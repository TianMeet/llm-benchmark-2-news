"""GenericTask answer_type 集成测试。

验收标准：
- 旧 YAML（无 answer_type）→ 行为不变
- 新 YAML（有 answer_type、无 metrics）→ 自动从 profile 取预设
- 混合 YAML（有 answer_type + 有 metrics）→ 显式 metrics 优先
"""

from __future__ import annotations

import pytest
import yaml
from pathlib import Path
from typing import Any

from bench.tasks.generic import GenericTask
from bench.task_metrics.profile import load_profile

# 真实任务 YAML 路径 — 用于验证向后兼容
IE_JSON_YAML = Path(__file__).resolve().parents[1] / "bench" / "tasks" / "ie_json.yaml"


@pytest.fixture
def write_yaml(tmp_path: Path):
    """将 dict 写入 tmp_path 下的临时 YAML 文件，pytest 自动清理。"""
    _counter = 0

    def _write(cfg: dict[str, Any]) -> str:
        nonlocal _counter
        _counter += 1
        p = tmp_path / f"task_{_counter}.yaml"
        p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        return str(p)

    return _write


class TestGenericTaskBackwardCompat:
    """旧 YAML 格式（无 answer_type）→ 行为完全不变。"""

    def test_existing_ie_json_loads(self) -> None:
        """现有 ie_json.yaml 仍可正常加载。"""
        task = GenericTask(IE_JSON_YAML)
        assert task.name == "ie_json"
        assert task._answer_type is None

    def test_existing_ie_json_metrics_unchanged(self) -> None:
        """无 answer_type 时 _metrics_cfg 保持原始值。"""
        task = GenericTask(IE_JSON_YAML)
        assert len(task._metrics_cfg) == 1
        assert task._metrics_cfg[0]["type"] == "field_completeness"

    def test_old_format_metrics_work(self, write_yaml) -> None:
        """旧格式的 metrics 计算仍正常运作。"""
        cfg = {
            "name": "test_old",
            "version": "v1",
            "prompt_template": "ie_json",
            "metrics": [
                {"type": "exact_match", "name": "acc",
                 "pred_field": "sentiment", "label_field": "gt_sentiment"},
            ],
        }
        task = GenericTask(write_yaml(cfg))
        result = task.metrics(
            sample={"gt_sentiment": "positive"},
            output="",
            parsed={"sentiment": "positive"},
        )
        assert result["acc"] == 1.0


class TestGenericTaskWithAnswerType:
    """新 YAML 格式（有 answer_type）。"""

    def setup_method(self) -> None:
        load_profile.cache_clear()

    def test_answer_type_attribute_set(self, write_yaml) -> None:
        """answer_type 应被正确读取。"""
        cfg = {
            "name": "test_choice",
            "version": "v1",
            "prompt_template": "ie_json",
            "answer_type": "choice",
        }
        task = GenericTask(write_yaml(cfg))
        assert task._answer_type == "choice"

    def test_answer_type_no_metrics_uses_preset(self, write_yaml) -> None:
        """有 answer_type 且无 metrics → 从 profile 取预设。"""
        cfg = {
            "name": "test_preset",
            "version": "v1",
            "prompt_template": "ie_json",
            "answer_type": "number",
            # 不写 metrics
        }
        task = GenericTask(write_yaml(cfg))
        assert len(task._metrics_cfg) > 0
        types = [m["type"] for m in task._metrics_cfg]
        assert "numeric_error" in types

    def test_answer_type_with_metrics_uses_task_cfg(self, write_yaml) -> None:
        """有 answer_type + 有 metrics → 显式配置优先。"""
        cfg = {
            "name": "test_override",
            "version": "v1",
            "prompt_template": "ie_json",
            "answer_type": "choice",
            "metrics": [
                {"type": "field_completeness", "fields": ["f1"]},
            ],
        }
        task = GenericTask(write_yaml(cfg))
        assert len(task._metrics_cfg) == 1
        assert task._metrics_cfg[0]["type"] == "field_completeness"

    def test_answer_type_preset_metrics_execute(self, write_yaml) -> None:
        """answer_type 自动获取的 metrics 可正常执行 metrics()。"""
        cfg = {
            "name": "test_exec",
            "version": "v1",
            "prompt_template": "ie_json",
            "answer_type": "choice",
            "parse_schema": [
                {"field": "value", "type": "str", "default": ""},
            ],
        }
        task = GenericTask(write_yaml(cfg))
        result = task.metrics(
            sample={"value": "positive"},
            output="",
            parsed={"value": "positive"},
        )
        # choice preset: exact_match → choice_acc
        assert "choice_acc" in result

    def test_metrics_method_unchanged_signature(self, write_yaml) -> None:
        """metrics() 返回 dict[str, Any]，不返回 ScoreCard。"""
        cfg = {
            "name": "test_sig",
            "version": "v1",
            "prompt_template": "ie_json",
            "answer_type": "choice",
        }
        task = GenericTask(write_yaml(cfg))
        result = task.metrics(sample={"value": "a"}, output="", parsed={"value": "a"})
        assert isinstance(result, dict)
