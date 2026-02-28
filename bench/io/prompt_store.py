"""
Prompt 模板存储器：按任务名懒加载 bench/prompts/*.yaml 模板。

委托 llm_core.prompt_renderer.PromptRenderer 实现 Jinja2 渲染，
本层仅负责模板文件发现与实例缓存，避免重复加载。

模板目录默认：bench/prompts/，与模块物理位置解耦。
"""

from __future__ import annotations

from pathlib import Path


class PromptStore:
    """从 `bench/prompts/*.yaml` 加载任务模板。"""

    def __init__(self, prompt_dir: str | Path | None = None):
        # 默认目录固定为 bench/prompts，与模块物理位置解耦。
        self.prompt_dir = Path(prompt_dir or Path(__file__).resolve().parent.parent / "prompts")
        self._renderers: dict[str, object] = {}

    def render(self, template_name: str, **kwargs) -> tuple[list[dict], str]:
        """渲染指定模板，返回 messages 与 prompt 版本号。"""
        if template_name not in self._renderers:
            # 懒加载避免无关任务提前引入外部依赖。
            from llm_core.prompt_renderer import PromptRenderer

            path = self.prompt_dir / f"{template_name}.yaml"
            self._renderers[template_name] = PromptRenderer(str(path), enable_few_shot=False)
        renderer = self._renderers[template_name]
        return renderer.render_with_version(**kwargs)
