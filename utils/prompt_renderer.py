"""
Prompt 渲染器 — 业务版（继承 llm_core 通用版，添加新闻/股票业务逻辑）
"""

import logging

from llm_core.prompt_renderer import PromptRenderer as _BasePromptRenderer

logger = logging.getLogger(__name__)


class PromptRenderer(_BasePromptRenderer):
    """业务 Prompt 渲染器：在通用渲染能力之上添加新闻内容截取和股票级渲染。"""

    def __init__(self, template_path: str, enable_few_shot: bool = True,
                 max_news_content_chars: int = 3000):
        super().__init__(template_path, enable_few_shot=enable_few_shot)
        self.max_news_content_chars = max_news_content_chars

    def render_stock(self, stock_data: dict) -> list[dict]:
        """
        渲染股票级综合分析 prompt（DWS 层使用）。

        Parameters
        ----------
        stock_data : dict
            包含以下键:
            - ticker_symbol, sec_short_name, exchange_cd, news_count
            - events: list[dict] — 事件列表（已排序、已截断）
            - dominant_event_type, event_diversity

        Returns
        -------
        list[dict]
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
        """
        return super().render(**stock_data)
