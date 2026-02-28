"""
LLM 响应解析器 — 通用 JSON 提取 + 字段校验工具（零业务逻辑）。

多策略解析管线（按优先级）：
  1. 全文直接解析/修复（json.loads → 单位归一化 → 注释移除 → 尾逗号 → Python 字面量回退）
  2. Markdown fenced code block 提取后重复步骤 1
  3. 括号平衡状态机提取首个完整 JSON 对象/数组

失败分类（通过 JsonParseMeta.reason）：
  ok / empty / truncated / invalid_json / non_string

设计参考：docs/design/robust_json_parser.md
"""

import ast
import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JsonParseMeta:
    """JSON 解析结果元信息。"""

    data: dict | None
    reason: str
    strategy: str


def _extract_first_balanced_json(text: str) -> tuple[str | None, bool]:
    """
    提取文本中第一个完整 JSON（对象或数组）。

    返回值: (candidate, truncated)
    - candidate: 提取到的 JSON 片段；未提取到时为 None
    - truncated: 检测到 JSON 起始但未闭合（疑似被截断）
    """
    stack: list[str] = []
    start = None
    in_string = False
    escape_next = False
    truncated = False
    pairs = {"}": "{", "]": "["}

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
            elif ch in "{[":
                if not stack:
                    start = i
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    continue
                if stack[-1] == pairs[ch]:
                    stack.pop()
                    if not stack and start is not None:
                        return text[start:i + 1], False
                else:
                    # 不匹配时重置，继续向后寻找下一个候选
                    stack.clear()
                    start = None
    if start is not None and stack:
        truncated = True
    return None, truncated


def _extract_first_fenced_block(text: str) -> str | None:
    """提取第一个 markdown 代码块内容。"""
    m = re.search(r"```(?:json|JSON|javascript|js|python|py)?\s*([\s\S]*?)```", text)
    if not m:
        return None
    return m.group(1).strip()


def _normalize_common_json_text(text: str) -> str:
    """归一化常见全角/智能引号符号，降低模型输出格式噪声。"""
    mapping = {
        "\ufeff": "",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "，": ",",
        "：": ":",
        "｛": "{",
        "｝": "}",
        "［": "[",
        "］": "]",
    }
    for src, dst in mapping.items():
        text = text.replace(src, dst)
    return text


def _strip_json_comments(text: str) -> str:
    """移除 JSON 外层的 // 与 /* */ 注释（字符串内保持不变）。"""
    out: list[str] = []
    i = 0
    in_string = False
    escape_next = False
    n = len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_string:
            out.append(ch)
            if escape_next:
                escape_next = False
            elif ch == "\\":
                escape_next = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            i += 2
            while i < n and text[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _remove_trailing_commas(text: str) -> str:
    """移除对象/数组闭合前的尾逗号。"""
    old = None
    new = text
    while old != new:
        old = new
        new = re.sub(r",\s*([}\]])", r"\1", new)
    return new


def _try_json_loads(candidate: str) -> object | None:
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _try_python_literal(candidate: str) -> object | None:
    """容忍单引号字典/列表等 Python 字面量风格输出。"""
    try:
        return ast.literal_eval(candidate)
    except Exception:
        return None


def _parse_candidate_as_object(candidate: str) -> dict | None:
    """把候选文本尽力解析为 dict。"""
    if not candidate.strip():
        return None

    # 1) 严格 JSON
    parsed = _try_json_loads(candidate)
    if isinstance(parsed, dict):
        return parsed

    # 2) 归一化 + 注释/尾逗号修复
    normalized = _remove_trailing_commas(_strip_json_comments(_normalize_common_json_text(candidate))).strip()
    parsed = _try_json_loads(normalized)
    if isinstance(parsed, dict):
        return parsed

    # 3) Python 字面量回退（单引号等）
    parsed = _try_python_literal(normalized)
    if isinstance(parsed, dict):
        return parsed
    return None


def parse_llm_json_with_meta(text: str) -> JsonParseMeta:
    """鲁棒 JSON 解析，返回结果 + 失败原因 + 命中策略。"""
    if not isinstance(text, str):
        return JsonParseMeta(data=None, reason="non_string", strategy="none")

    raw = _normalize_common_json_text(text).strip()
    if not raw:
        return JsonParseMeta(data=None, reason="empty", strategy="none")

    # 策略 1: 全文直接解析/修复
    obj = _parse_candidate_as_object(raw)
    if obj is not None:
        return JsonParseMeta(data=obj, reason="ok", strategy="full_text")

    # 策略 2: 提取 markdown fenced code block
    fenced = _extract_first_fenced_block(raw)
    if fenced:
        obj = _parse_candidate_as_object(fenced)
        if obj is not None:
            return JsonParseMeta(data=obj, reason="ok", strategy="fenced_block")

    # 策略 3: 括号平衡提取首个 JSON 片段
    balanced, truncated = _extract_first_balanced_json(raw)
    if balanced:
        obj = _parse_candidate_as_object(balanced)
        if obj is not None:
            return JsonParseMeta(data=obj, reason="ok", strategy="balanced_extract")

    if truncated:
        return JsonParseMeta(data=None, reason="truncated", strategy="balanced_extract")
    return JsonParseMeta(data=None, reason="invalid_json", strategy="none")


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
    return parse_llm_json_with_meta(text).data


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
