from llm_core.response_parser import parse_llm_json, parse_llm_json_with_meta


def test_parse_llm_json_parses_markdown_code_block() -> None:
    text = "Here is result:\n```json\n{\"a\": 1, \"b\": \"x\"}\n```\nthanks"
    parsed = parse_llm_json(text)
    assert parsed == {"a": 1, "b": "x"}


def test_parse_llm_json_repairs_trailing_comma() -> None:
    text = '{"a": 1, "b": [1,2,],}'
    parsed = parse_llm_json(text)
    assert parsed == {"a": 1, "b": [1, 2]}


def test_parse_llm_json_repairs_python_literal_dict() -> None:
    text = "{'a': 1, 'b': 'ok', 'flag': True}"
    parsed = parse_llm_json(text)
    assert parsed == {"a": 1, "b": "ok", "flag": True}


def test_parse_llm_json_repairs_comments_and_fullwidth_punct() -> None:
    text = "｛\"a\":1，/*note*/\"b\":2｝"
    parsed = parse_llm_json(text)
    assert parsed == {"a": 1, "b": 2}


def test_parse_llm_json_with_meta_truncated() -> None:
    text = '{"a":"hello"'
    meta = parse_llm_json_with_meta(text)
    assert meta.data is None
    assert meta.reason == "truncated"


def test_parse_llm_json_with_meta_empty() -> None:
    meta = parse_llm_json_with_meta("   \n\t")
    assert meta.data is None
    assert meta.reason == "empty"


def test_parse_llm_json_with_meta_invalid() -> None:
    meta = parse_llm_json_with_meta("totally not json")
    assert meta.data is None
    assert meta.reason == "invalid_json"
