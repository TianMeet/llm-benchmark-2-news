from pathlib import Path

from bench.registry import ModelRegistry
from llm_core.config import load_yaml


def test_load_yaml_reads_dotenv_in_same_directory(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "cfg.yaml"
    dotenv_path = tmp_path / ".env"
    config_path.write_text("api_key: ${CAIJJ_API_KEY}\n", encoding="utf-8")
    dotenv_path.write_text("CAIJJ_API_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.delenv("CAIJJ_API_KEY", raising=False)

    cfg = load_yaml(config_path)
    assert cfg["api_key"] == "from_dotenv"


def test_dotenv_does_not_override_existing_env(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "cfg.yaml"
    dotenv_path = tmp_path / ".env"
    config_path.write_text("api_key: ${CAIJJ_API_KEY}\n", encoding="utf-8")
    dotenv_path.write_text("CAIJJ_API_KEY=from_dotenv\n", encoding="utf-8")
    monkeypatch.setenv("CAIJJ_API_KEY", "from_env")

    cfg = load_yaml(config_path)
    assert cfg["api_key"] == "from_env"


def test_model_registry_json_reads_sibling_dotenv(tmp_path, monkeypatch) -> None:
    reg_path = tmp_path / "registry.json"
    dotenv_path = tmp_path / ".env"
    monkeypatch.delenv("MY_TEST_KEY", raising=False)
    dotenv_path.write_text("MY_TEST_KEY=from_dotenv\n", encoding="utf-8")
    reg_path.write_text(
        '{"active":"m1","models":{"m1":{"provider":"openai_compatible","model":"x","api_key":"${MY_TEST_KEY}","base_url":"http://localhost"}}}',
        encoding="utf-8",
    )
    registry = ModelRegistry(reg_path)
    cfg = registry.resolve_config("m1")
    assert cfg["api_key"] == "from_dotenv"
