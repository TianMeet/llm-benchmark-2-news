"""评测任务协议定义。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class EvalTask(ABC):
    """任务契约：构建 prompt、解析输出、计算任务指标。"""

    name: str = ""     # 子类必须赋值为非空任务名
    version: str = "v1"  # 子类可覆盖以声明版本
    parse_applicable: bool = True

    @abstractmethod
    def build_prompt(self, sample: dict[str, Any], context: dict[str, Any] | None = None) -> tuple[list[dict], str]:
        """构建 LLM 输入 messages，并返回 prompt_version。"""

    @abstractmethod
    def parse(self, output: str) -> dict[str, Any] | None:
        """将模型原始文本解析为结构化结果。"""

    @abstractmethod
    def metrics(self, sample: dict[str, Any], output: str, parsed: dict[str, Any] | None) -> dict[str, Any]:
        """计算任务特定指标。"""

    def mock_response(self, sample: dict[str, Any]) -> str:
        """返回离线 mock 模式的响应内容字符串。

        默认实现返回空 JSON 对象；子类可覆盖以返回任务特定的合理默认值。
        子类（如 GenericTask）应从 YAML 中读取 mock_response 字段以使 Gateway
        无需硬编码任务名称判断。
        """
        return json.dumps({})
