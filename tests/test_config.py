from pathlib import Path

from evalguard.config import load_settings


def test_default_settings_loaded(tmp_path: Path) -> None:
    settings = load_settings()
    assert settings.rag.retriever_top_k == 4
    assert "mock" in settings.providers

    config_path = tmp_path / "config.yaml"
    config_path.write_text("rag:\n  retriever_top_k: 2\n", encoding="utf-8")
    settings = load_settings(config_path)
    assert settings.rag.retriever_top_k == 2

