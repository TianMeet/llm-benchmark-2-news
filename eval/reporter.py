"""报告器抽象与内置实现。

Reporter 定义通用报告接口，内置提供 MarkdownReporter 实现。
加入新承载形式（如 HTMLReporter）只需继承 Reporter 并覆盖 generate()。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Reporter(ABC):
    """报告生成抽象接口，便于后续扩展为 HTML、JSON 等多种输出格式。"""

    @abstractmethod
    def generate(self, run_dir: str | Path) -> Path:
        """根据 run_dir 产出报告文件，返回报告文件路径。"""
        ...


class MarkdownReporter(Reporter):
    """生成 Markdown 格式运行报告的内置实现。"""

    def generate(self, run_dir: str | Path) -> Path:
        from eval.report import generate_report
        return generate_report(run_dir)
