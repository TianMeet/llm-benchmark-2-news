"""报告器抽象与内置实现。

Reporter 定义通用报告接口，当前内置 MarkdownReporter。
后续可扩展为 HTML、JSON 等多种输出格式，只需继承并覆写 generate()。

MarkdownReporter 委托 bench/reporting/report.py 中的 generate_report() 实现，
输出内容包括：
- 总表（Model x Metric）
- 按任务维度模型对比（含 tm_* 任务指标）
- Workflow 维度概览
- 失败类型分布 + 失败样本摘要（Top 5）
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
        from bench.reporting.report import generate_report
        return generate_report(run_dir)
