"""数据驱动通用任务：通过 YAML 配置文件完整定义一个评测任务，无需写 Python 代码。

任务 YAML 结构：
  name: str
  version: str
  prompt_template: str          # bench/prompts/ 下的模板文件名（不含 .yaml）
  serialize_fields:             # sample 中需要序列化为 JSON 字符串的字段（可选）
    - news_items                # 会以 news_items_json 注入模板
  parse_schema:                 # 输出解析规则列表（可选，不填则 parse_applicable=False）
    - field: summary
      type: str                 # str / int / float / enum / list / list_raw
      default: ""
      values: []                # 仅 enum 生效
      lo: null                  # 仅 int/float 生效
      hi: null
  metrics:                      # 指标规则列表（可选）
    - type: field_completeness   # 可选：field_completeness / reference_rouge / custom_ratio
      fields: [f1, f2]
    - type: reference_rouge      # 需 sample 中有 reference_summary 字段与 rouge 包
      output_field: summary
      reference_field: reference_summary
    - type: custom_ratio         # 任意分子/分母字段之比
      name: dedup_ratio
      numerator_field: removed_count
      denominator_sample_field: news_items   # 取 len(sample[field])
    - type: exact_match          # 分类/标签准确率（0/1）
      name: sentiment_acc
      pred_field: sentiment
      label_field: gt_sentiment
    - type: numeric_error        # 数值误差（MAE）+ 容差命中率
      name: impact
      pred_field: impact_score
      label_field: gt_impact_score
      tolerance: 1
    - type: list_overlap         # 列表重叠（precision/recall/f1）
      name: keyword
      pred_field: keywords
      label_field: gt_keywords
    - type: reasoning_consistency   # final answer 与推理文本一致性
      name: reasoning
      final_answer_field: final_answer
      support_fields: [steps, claims, rationale]
    - type: claim_evidence_overlap  # claim 引用证据与标注证据重叠
      name: evidence
      claims_field: claims
      label_field: gt_evidence_ids
      claim_evidence_key: evidence
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from bench.io.prompt_store import PromptStore
from bench.task_metrics import (
    compute_metric,
)
from bench.tasks.base import EvalTask
from llm_core.response_parser import (
    parse_llm_json,
    validate_enum_field,
    validate_int_range,
    validate_list_field,
    validate_str_field,
)

logger = logging.getLogger(__name__)


class GenericTask(EvalTask):
    """由 YAML 配置文件完整驱动的通用任务，支持任意 sample 字段组合。"""

    def __init__(self, task_config_path: str | Path) -> None:
        cfg_path = Path(task_config_path)
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.name: str = str(cfg["name"])
        self.version: str = str(cfg.get("version", "v1"))
        self._prompt_template: str = str(cfg["prompt_template"])
        self._serialize_fields: list[str] = list(cfg.get("serialize_fields", []))
        self._parse_schema: list[dict] = list(cfg.get("parse_schema", []))
        self._metrics_cfg: list[dict] = list(cfg.get("metrics", []))
        self.default_params: dict[str, Any] = dict(cfg.get("default_params") or {})
        self.parse_applicable: bool = bool(self._parse_schema)
        # mock_response 由 YAML 声明，使 Gateway 无需硬编码任务名称判断（P1 解耦）
        self._mock_response_data: dict | None = cfg.get("mock_response")
        # field_mapping：将数据集字段名映射为任务内规范字段名。
        # 格式：{dataset_field: canonical_field}
        # 映射结果叠加在原始 sample 之上（非破坏性），原字段保留。
        self._field_mapping: dict[str, str] = dict(cfg.get("field_mapping") or {})

        self._prompts: PromptStore | None = None

    def _resolve_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """将 field_mapping 应用于 sample，返回含规范字段名的新字典。

        - 原始字段全部保留（不破坏现有逻辑）
        - 映射后的规范字段名以覆盖方式叠加
        - 若 YAML 未配置 field_mapping，直接返回原对象（零开销）
        """
        if not self._field_mapping:
            return sample
        resolved = dict(sample)
        for src, dst in self._field_mapping.items():
            if src in sample:
                resolved[dst] = sample[src]
        return resolved

    # ── Prompt ──────────────────────────────────────────────────────────

    def mock_response(self, sample: dict[str, Any]) -> str:
        """返回离线 mock 模式的响应内容字符串。

        若 YAML 中声明了 mock_response 字段则直接序列化返回；
        否则回退到基类的空 JSON 对象\u3002
        """
        if self._mock_response_data is not None:
            return json.dumps(self._mock_response_data, ensure_ascii=False)
        return super().mock_response(sample)

    def build_prompt(
        self,
        sample: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> tuple[list[dict], str]:
        """把整个 sample 展开为 Jinja2 变量，serialize_fields 自动序列化为 JSON 字符串。

        优先级：context["input"][field] > sample[field] > []
        workflow 模式下上一步解析结果通过 context["input"] 传入。
        """
        if self._prompts is None:
            self._prompts = PromptStore()

        ctx = context or {}
        # workflow 模式下上一步输出作为当前步骤的逻辑输入
        ctx_input: dict[str, Any] = ctx.get("input") or {}
        template_vars: dict[str, Any] = {}

        # 1. sample 所有字段注入（先应用 field_mapping，使模板可直接使用规范字段名）
        template_vars.update(self._resolve_sample(sample))

        # 2. context 额外变量叠加
        template_vars.update(ctx)

        # 3. serialize_fields：优先取 context["input"] 中的字段，再取 sample，最后兜底空列表
        for field in self._serialize_fields:
            val = ctx_input.get(field, sample.get(field, []))
            template_vars[f"{field}_json"] = json.dumps(val, ensure_ascii=False)

        return self._prompts.render(self._prompt_template, **template_vars)

    # ── Parse ───────────────────────────────────────────────────────────

    def parse(self, output: str) -> dict[str, Any] | None:
        """按 parse_schema 定义的规则解析并校验模型输出。"""
        if not self._parse_schema:
            return None

        data = parse_llm_json(output)
        if not isinstance(data, dict):
            return None

        result: dict[str, Any] = {}
        for rule in self._parse_schema:
            field = rule["field"]
            ftype = rule.get("type", "str")
            default = rule.get("default", "" if ftype == "str" else 0)

            if ftype == "str":
                result[field] = validate_str_field(data, field, str(default))
            elif ftype == "int":
                lo = rule.get("lo", -(10 ** 9))
                hi = rule.get("hi", 10 ** 9)
                result[field] = validate_int_range(data, field, int(default), int(lo), int(hi))
            elif ftype == "float":
                raw = data.get(field, default)
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    val = float(default)
                lo = rule.get("lo")
                hi = rule.get("hi")
                if lo is not None:
                    val = max(float(lo), val)
                if hi is not None:
                    val = min(float(hi), val)
                result[field] = val
            elif ftype == "enum":
                values = set(rule.get("values", []))
                result[field] = validate_enum_field(data, field, values, str(default), log_fix=True)
            elif ftype == "list":
                max_len = rule.get("max_len")
                result[field] = validate_list_field(data, field, default=[], max_len=max_len)
            elif ftype == "list_raw":
                raw = data.get(field, default if isinstance(default, list) else [])
                result[field] = raw if isinstance(raw, list) else []
            else:
                result[field] = data.get(field, default)

        return result

    # ── Metrics ─────────────────────────────────────────────────────────

    def metrics(
        self,
        sample: dict[str, Any],
        output: str,
        parsed: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """按 metrics 配置计算指标，分发到 bench.task_metrics.compute_metric。"""
        # 应用 field_mapping，使 label_field / reference_field 等可使用规范字段名。
        sample = self._resolve_sample(sample)
        result: dict[str, Any] = {}
        for rule in self._metrics_cfg:
            mtype = rule.get("type", "")
            result.update(compute_metric(mtype, rule, sample, parsed, task_name=self.name))
        return result
