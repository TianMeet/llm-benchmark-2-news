"""Stock scoring task outputting score and rationale."""

from __future__ import annotations

import json
from typing import Any

from llm_core.response_parser import parse_llm_json, validate_int_range, validate_str_field

from eval.prompt_store import PromptStore
from eval.tasks.base import EvalTask


class StockScoreTask(EvalTask):
    name = "stock_score"
    version = "v1"

    def __init__(self) -> None:
        self.prompts: PromptStore | None = None

    def build_prompt(self, sample: dict[str, Any], context: dict[str, Any] | None = None) -> tuple[list[dict], str]:
        if self.prompts is None:
            self.prompts = PromptStore()
        context = context or {}
        input_payload = context.get("input", sample)
        news_json = json.dumps(input_payload.get("news_items", sample.get("news_items", [])), ensure_ascii=False)
        return self.prompts.render(
            "stock_score",
            ticker=sample.get("ticker", "UNKNOWN"),
            news_items_json=news_json,
        )

    def parse(self, output: str) -> dict[str, Any] | None:
        data = parse_llm_json(output)
        if not isinstance(data, dict):
            return None
        return {
            "score": validate_int_range(data, "score", default=0, lo=-100, hi=100),
            "rationale": validate_str_field(data, "rationale", ""),
        }

    def metrics(self, sample: dict[str, Any], output: str, parsed: dict[str, Any] | None) -> dict[str, Any]:
        if not parsed:
            return {"score_valid": 0, "score_value": 0}
        return {
            "score_valid": int(-100 <= int(parsed["score"]) <= 100),
            "score_value": int(parsed["score"]),
        }
