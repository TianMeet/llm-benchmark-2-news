"""通用 Prompt 渲染器 — Jinja2 模板加载与渲染（零业务依赖）。

YAML 模板结构：
  version: "v3"
  system: "..."           # Jinja2 模板（不含行级变量，预编译缓存）
  user: "..."             # Jinja2 模板（含 sample 字段变量）
  few_shot_examples:      # 可选 few-shot 示例（预编译缓存）
    - user: "..."
      assistant: "..."

输出格式：[{"role": "system", ...}, {"role": "user", ...}]
性能优化：system prompt 和 few-shot 在实例化时预编译缓存，
          仅 user prompt 每次渲染。
"""

import logging
from pathlib import Path

import yaml
from jinja2 import BaseLoader, Environment

logger = logging.getLogger(__name__)


class PromptRenderer:
    """加载 YAML prompt 模板，渲染为 LLM messages 列表。"""

    def __init__(self, template_path: str, enable_few_shot: bool = True):
        self.template_path = Path(template_path)
        self.enable_few_shot = enable_few_shot

        with open(self.template_path, "r", encoding="utf-8") as f:
            self.template_data = yaml.safe_load(f)

        self.version = self.template_data.get("version", "unknown")
        self.jinja_env = Environment(loader=BaseLoader(), keep_trailing_newline=True)

        # 预渲染 system prompt（不含行级 Jinja2 变量）
        self._cached_system_content = self.jinja_env.from_string(
            self.template_data["system"]
        ).render()

        # 预构建 few-shot messages
        self._cached_few_shot_messages = []
        if self.enable_few_shot:
            for ex in self.template_data.get("few_shot_examples", []):
                self._cached_few_shot_messages.append(
                    {"role": "user", "content": ex["user"].strip()}
                )
                self._cached_few_shot_messages.append(
                    {"role": "assistant", "content": ex["assistant"].strip()}
                )

        # 预编译 user template
        self._user_template = self.jinja_env.from_string(self.template_data["user"])

        logger.info("Loaded prompt template: %s (version=%s)", self.template_path, self.version)

    def render(self, **kwargs) -> list[dict]:
        """
        使用任意关键字参数渲染完整的 messages 列表。

        Parameters
        ----------
        **kwargs
            传给 Jinja2 user template 的变量。

        Returns
        -------
        list[dict]
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
        """
        messages = []

        # System message（使用缓存）
        messages.append({"role": "system", "content": self._cached_system_content})

        # Few-shot examples（使用缓存）
        messages.extend(self._cached_few_shot_messages)

        # User message — 渲染带实际数据的 user prompt
        user_content = self._user_template.render(**kwargs)
        messages.append({"role": "user", "content": user_content})

        return messages

    def render_with_version(self, **kwargs) -> tuple[list[dict], str]:
        """渲染 messages 并返回 ``(messages, prompt_version)`` 元组。

        等价于 ``(self.render(**kwargs), self.version)``，方便与
        ``build_prompt`` 等需要 ``tuple[list[dict], str]`` 的接口保持一致。
        """
        return self.render(**kwargs), str(self.version)
