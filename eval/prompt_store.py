"""Prompt template registry for eval tasks."""

from __future__ import annotations

from pathlib import Path


class PromptStore:
    """Loads task prompt templates from eval/prompts/*.yaml."""

    def __init__(self, prompt_dir: str | Path | None = None):
        self.prompt_dir = Path(prompt_dir or Path(__file__).resolve().parent / "prompts")
        self._renderers: dict[str, object] = {}

    def render(self, template_name: str, **kwargs) -> tuple[list[dict], str]:
        if template_name not in self._renderers:
            from llm_core.prompt_renderer import PromptRenderer

            path = self.prompt_dir / f"{template_name}.yaml"
            self._renderers[template_name] = PromptRenderer(str(path), enable_few_shot=False)
        renderer = self._renderers[template_name]
        return renderer.render(**kwargs), str(renderer.version)
