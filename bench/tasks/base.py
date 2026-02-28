"""评测任务抽象基类——所有任务的契约接口。

定义了评测任务必须实现的三个抽象方法：
1. build_prompt()：构建 LLM 输入 messages
2. parse()：将模型原始文本解析为结构化结果
3. metrics()：计算任务特定指标

当前项目中所有任务均通过 GenericTask（YAML 驱动）实现，
无需直接继承此基类。但特殊任务仍可用 Python 继承。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class EvalTask(ABC):
    """任务契约：构建 prompt、解析输出、计算任务指标。"""

    name: str = ""     # 子类必须赋值为非空任务名，对应 bench/tasks/{name}.yaml
    version: str = "v1"  # 任务版本，记录在 results.jsonl 中用于回归追踪
    parse_applicable: bool = True   # 是否需要解析模型输出；False 时 parse_success 不影响评分
    default_params: dict[str, Any] = {}  # 任务级默认模型参数，如 response_format

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
