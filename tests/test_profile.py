"""bench/task_metrics/profile.py 单元测试。

验收标准：
- load_profile() 能从真实 YAML 正确加载 ScoringProfile
- load_profile() lru_cache 生效（同路径返回同一对象）
- _validate_profile() 在 metric type 未注册时输出 WARNING
- resolve_metrics() 三层合并策略正确：
  1. answer_type=None → 返回原始 task_metrics_cfg
  2. answer_type 有值 + cfg 为空 → 返回 profile 预设
  3. answer_type 有值 + cfg 非空 → 返回任务级 cfg
  4. answer_type 在 profile 中不存在 → 回退到 task_metrics_cfg
"""

from __future__ import annotations

import logging
import pytest
from pathlib import Path
from typing import Any

from conftest import SCORING_PROFILE_PATH as PROFILE_PATH

from bench.contracts.scoring import (
    MetricRule,
    NormalizationConfig,
    ScorerPreset,
    ScoringProfile,
)
from bench.task_metrics.profile import _validate_profile, load_profile, resolve_metrics


# ── load_profile ─────────────────────────────────────────────────────

class TestLoadProfile:
    def setup_method(self) -> None:
        """每个测试前清空 lru_cache。"""
        load_profile.cache_clear()

    def test_load_real_yaml(self) -> None:
        """加载真实 scoring_profile.template.yaml 成功。"""
        profile = load_profile(str(PROFILE_PATH))
        assert profile.version == "v1"
        assert "choice" in profile.scorers
        assert "number" in profile.scorers
        assert "list" in profile.scorers
        assert "text" in profile.scorers
        assert "reasoning" in profile.scorers

    def test_load_default_path(self) -> None:
        """不传参数时加载默认路径。"""
        profile = load_profile()
        assert profile.version == "v1"

    def test_lru_cache_same_object(self) -> None:
        """同一路径返回缓存对象（同一引用）。"""
        p1 = load_profile(str(PROFILE_PATH))
        p2 = load_profile(str(PROFILE_PATH))
        assert p1 is p2

    def test_nonexistent_file_raises(self) -> None:
        """不存在的文件抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_profile("/nonexistent/path.yaml")

    def test_preset_has_pass_threshold(self) -> None:
        """每个 preset 都应有 pass_threshold。"""
        profile = load_profile(str(PROFILE_PATH))
        for atype, preset in profile.scorers.items():
            assert isinstance(preset.pass_threshold, float), f"{atype} 缺少 pass_threshold"
            assert 0.0 <= preset.pass_threshold <= 1.0, f"{atype} pass_threshold 越界"


# ── _validate_profile ────────────────────────────────────────────────

class TestValidateProfile:
    def test_warns_on_unknown_metric_type(self, caplog: pytest.LogCaptureFixture) -> None:
        """未注册的 metric type 应产生 WARNING。"""
        profile = ScoringProfile(
            version="v1",
            normalization=NormalizationConfig(),
            scorers={
                "test_type": ScorerPreset(
                    primary_metric="fake_metric",
                    pass_threshold=0.5,
                    metrics=(MetricRule(type="nonexistent_calc"),),
                ),
            },
        )
        with caplog.at_level(logging.WARNING):
            _validate_profile(profile)
        assert "nonexistent_calc" in caplog.text

    def test_no_warning_for_registered_types(self, caplog: pytest.LogCaptureFixture) -> None:
        """已注册的 metric type 不应产生 WARNING。"""
        profile = ScoringProfile(
            version="v1",
            normalization=NormalizationConfig(),
            scorers={
                "choice": ScorerPreset(
                    primary_metric="exact_match",
                    pass_threshold=0.5,
                    metrics=(MetricRule(type="exact_match"),),
                ),
            },
        )
        with caplog.at_level(logging.WARNING):
            _validate_profile(profile)
        assert "未注册" not in caplog.text


# ── resolve_metrics ──────────────────────────────────────────────────

class TestResolveMetrics:
    def setup_method(self) -> None:
        load_profile.cache_clear()

    def test_none_answer_type_returns_task_cfg(self) -> None:
        """answer_type=None → 直接返回 task_metrics_cfg（向后兼容）。"""
        task_cfg = [{"type": "exact_match", "pred_field": "a", "label_field": "b"}]
        result = resolve_metrics(answer_type=None, task_metrics_cfg=task_cfg)
        assert result is task_cfg  # 同一引用，零开销

    def test_answer_type_empty_cfg_uses_preset(self) -> None:
        """answer_type 有值 + 空 metrics → 从 profile 取预设。"""
        result = resolve_metrics(
            answer_type="choice",
            task_metrics_cfg=[],
            profile_path=str(PROFILE_PATH),
        )
        assert len(result) > 0
        assert result[0]["type"] == "exact_match"

    def test_answer_type_with_cfg_uses_task_cfg(self) -> None:
        """answer_type 有值 + 非空 metrics → 任务级 cfg 优先。"""
        task_cfg = [{"type": "field_completeness", "fields": ["f1"]}]
        result = resolve_metrics(
            answer_type="choice",
            task_metrics_cfg=task_cfg,
            profile_path=str(PROFILE_PATH),
        )
        assert result is task_cfg

    def test_unknown_answer_type_falls_back(self, caplog: pytest.LogCaptureFixture) -> None:
        """answer_type 不在 profile 中 → 回退到 task_metrics_cfg 并 WARNING。"""
        task_cfg = [{"type": "exact_match"}]
        with caplog.at_level(logging.WARNING):
            result = resolve_metrics(
                answer_type="unknown_type",
                task_metrics_cfg=task_cfg,
                profile_path=str(PROFILE_PATH),
            )
        assert result is task_cfg
        assert "unknown_type" in caplog.text

    def test_preset_metrics_have_type_field(self) -> None:
        """profile 预设返回的每个 metric dict 都必须有 'type' 字段。"""
        for answer_type in ("choice", "number", "list", "text", "reasoning"):
            result = resolve_metrics(
                answer_type=answer_type,
                task_metrics_cfg=[],
                profile_path=str(PROFILE_PATH),
            )
            for metric in result:
                assert "type" in metric, f"{answer_type} 预设中有 metric 缺少 type"
