"""Structured extraction task requiring JSON output."""

from __future__ import annotations

import json
from typing import Any

from llm_core.response_parser import (
    parse_llm_json,
    validate_enum_field,
    validate_int_range,
    validate_str_field,
)

from eval.prompt_store import PromptStore
from eval.tasks.base import EvalTask


class IEJsonTask(EvalTask):
    name = "ie_json"
    version = "v1"

    def __init__(self) -> None:
        self.prompts: PromptStore | None = None

    def build_prompt(self, sample: dict[str, Any], context: dict[str, Any] | None = None) -> tuple[list[dict], str]:
        if self.prompts is None:
            self.prompts = PromptStore()
        news_json = json.dumps(sample.get("news_items", []), ensure_ascii=False)
        return self.prompts.render(
            "ie_json",
            ticker=sample.get("ticker", "UNKNOWN"),
            news_items_json=news_json,
        )

    def parse(self, output: str) -> dict[str, Any] | None:
        data = parse_llm_json(output)
        if not isinstance(data, dict):
            return None
        return {
            "ticker": validate_str_field(data, "ticker", ""),
            "event_type": validate_enum_field(
                data,
                "event_type",
                {"earnings", "policy", "product", "macro", "other"},
                "other",
                log_fix=False,
            ),
            "sentiment": validate_enum_field(
                data,
                "sentiment",
                {"positive", "neutral", "negative"},
                "neutral",
                log_fix=False,
            ),
            "impact_score": validate_int_range(data, "impact_score", default=0, lo=-5, hi=5),
            "summary": validate_str_field(data, "summary", ""),
        }

    def metrics(self, sample: dict[str, Any], output: str, parsed: dict[str, Any] | None) -> dict[str, Any]:
        if not parsed:
            return {"field_completeness": 0.0}
        required = ["ticker", "event_type", "sentiment", "impact_score", "summary"]
        non_empty = 0
        for k in required:
            val = parsed.get(k)
            if isinstance(val, str):
                non_empty += int(bool(val.strip()))
            else:
                non_empty += int(val is not None)
        return {"field_completeness": round(non_empty / len(required), 4)}
