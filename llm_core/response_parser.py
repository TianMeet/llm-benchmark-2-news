"""
LLM 响应解析器 — 通用 JSON 提取 + 字段校验工具（零业务逻辑）
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


def _extract_first_json_object(text: str) -> str | None:
    """
    通过追踪 {} 括号深度，提取文本中第一个完整的 JSON 对象。

    感知字符串上下文：在双引号内的花括号不影响深度计数，
    避免 JSON 值中包含 { } 时导致提取截断。

    使用 escape_next 状态机正确处理连续转义（如 \\\\" 中 \\\\ 是转义反斜杠，
    后面的 " 是真正的引号结束符）。
    """
    depth = 0
    start = None
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if in_string:
            if escape_next:
                escape_next = False
            elif ch == '\\':
                escape_next = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
                escape_next = False
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start:i+1]
    return None


def parse_llm_json(text: str) -> dict | None:
    """
    从 LLM 输出文本中提取 JSON 对象。

    尝试策略（按优先级）：
    1. 直接 json.loads(text.strip())
    2. 括号平衡提取第一个完整 JSON 对象
    3. 去掉 markdown 代码块标记后再解析

    Returns
    -------
    dict | None
        解析成功返回 dict，全部失败返回 None
    """
    text = text.strip()

    # 策略 1: 直接解析
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 策略 2: 括号平衡提取第一个完整 JSON 对象
    json_str = _extract_first_json_object(text)
    if json_str:
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

    # 策略 3: 去掉 markdown 代码块标记
    cleaned = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    cleaned = re.sub(r'\s*```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    return None


# ======================================================================
# 通用字段校验工具
# ======================================================================

def validate_str_field(data: dict, key: str, default: str = "") -> str:
    """验证字符串字段：非空时转 str，否则返回默认值。"""
    val = data.get(key, default)
    return str(val) if val else default


def validate_list_field(
    data: dict, key: str, default: list | None = None, max_len: int | None = None,
) -> list[str]:
    """验证列表字段：非 list 时返回空列表，元素转 str，可选截断。"""
    val = data.get(key, default if default is not None else [])
    if not isinstance(val, list):
        val = []
    result = [str(e) for e in val]
    return result[:max_len] if max_len else result


def validate_enum_field(
    data: dict, key: str, valid_set: set, default: str, log_fix: bool = True,
) -> str:
    """验证枚举字段：不在合法集合中时回退到默认值。"""
    val = data.get(key, default)
    if val not in valid_set:
        if log_fix:
            logger.info("field_fix: %s '%s' -> '%s'", key, val, default)
        return default
    return val


def validate_int_range(
    data: dict, key: str, default: int, lo: int, hi: int,
) -> int:
    """验证整数范围字段：转 int 失败用默认值，然后 clamp 到 [lo, hi]。"""
    val = data.get(key, default)
    try:
        val = int(val)
    except (TypeError, ValueError):
        val = default
    return max(lo, min(hi, val))
