"""News deduplication helper task for workflow step."""

from __future__ import annotations

import json
from typing import Any

from llm_core.response_parser import parse_llm_json, validate_int_range, validate_list_field

from eval.prompt_store import PromptStore
from eval.tasks.base import EvalTask


class NewsDedupTask(EvalTask):
    name = "news_dedup"
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
            "news_dedup",
            ticker=sample.get("ticker", "UNKNOWN"),
            news_items_json=news_json,
        )

    def parse(self, output: str) -> dict[str, Any] | None:
        data = parse_llm_json(output)
        if not isinstance(data, dict):
            return None
        return {
            "deduped_titles": validate_list_field(data, "deduped_titles", default=[]),
            "removed_count": validate_int_range(data, "removed_count", default=0, lo=0, hi=1000),
        }

    def metrics(self, sample: dict[str, Any], output: str, parsed: dict[str, Any] | None) -> dict[str, Any]:
        if not parsed:
            return {"dedup_ratio": 0.0}
        original = max(1, len(sample.get("news_items", [])))
        kept = len(parsed.get("deduped_titles", []))
        return {"dedup_ratio": round((original - kept) / original, 4)}
