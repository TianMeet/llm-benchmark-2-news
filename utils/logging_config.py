"""
统一日志配置模块 (utils/logging_config.py)
==========================================

用法:
    from utils.logging_config import setup_logging

    logger = setup_logging(__name__)
"""

import logging

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """配置全局日志格式并返回指定名称的 logger。"""
    logging.basicConfig(level=level, format=LOG_FORMAT)
    return logging.getLogger(name)
