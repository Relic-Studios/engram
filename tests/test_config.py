"""Tests for engram.core.config."""

import pytest
from pathlib import Path
from engram.core.config import Config


class TestConfig:
    def test_defaults(self):
        c = Config()
        assert c.signal_mode == "hybrid"
        assert c.extract_mode == "off"
        assert c.llm_provider == "ollama"
        assert c.token_budget == 12000

    def test_from_data_dir(self, tmp_path):
        c = Config.from_data_dir(tmp_path)
        assert c.data_dir == tmp_path.resolve()
        assert c.db_path == tmp_path.resolve() / "engram.db"

    def test_derived_paths(self, tmp_path):
        c = Config.from_data_dir(tmp_path)
        assert c.soul_dir == tmp_path.resolve() / "soul"
        assert c.semantic_dir == tmp_path.resolve() / "semantic"
        assert c.procedural_dir == tmp_path.resolve() / "procedural"
        assert c.embeddings_dir == tmp_path.resolve() / "embeddings"
        assert c.soul_path == tmp_path.resolve() / "soul" / "SOUL.md"
        assert c.identities_path == tmp_path.resolve() / "semantic" / "identities.yaml"

    def test_ensure_directories(self, tmp_path):
        c = Config.from_data_dir(tmp_path)
        c.ensure_directories()
        assert c.soul_dir.is_dir()
        assert c.semantic_dir.is_dir()
        assert c.procedural_dir.is_dir()
        assert c.embeddings_dir.is_dir()

    def test_from_yaml(self, tmp_path):
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(
            f"engram:\n"
            f"  data_dir: {tmp_path}\n"
            f"  signal_mode: regex\n"
            f"  token_budget: 4000\n",
            encoding="utf-8",
        )
        c = Config.from_yaml(yaml_path)
        assert c.signal_mode == "regex"
        assert c.token_budget == 4000

    def test_from_yaml_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Config.from_yaml(tmp_path / "nonexistent.yaml")

    def test_to_dict(self):
        c = Config()
        d = c.to_dict()
        assert "data_dir" in d
        assert "signal_mode" in d
        assert d["signal_mode"] == "hybrid"

    def test_custom_llm_func(self):
        def my_func(system, user):
            return "ok"

        c = Config(llm_func=my_func)
        assert c.get_llm_func() is my_func

    def test_custom_provider_no_func(self):
        c = Config(llm_provider="custom")
        with pytest.raises(ValueError, match="no llm_func was provided"):
            c.get_llm_func()

    def test_unknown_provider(self):
        c = Config(llm_provider="alien")
        with pytest.raises(ValueError, match="Unknown llm_provider"):
            c.get_llm_func()
