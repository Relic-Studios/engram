"""Tests for the dual-embedding system (A2: code-optimized embeddings).

Tests cover:
  - Content classification (is_code_content)
  - CodeEmbedder class (lazy loading, encode, fallback)
  - build_code_embedding_func factory
  - SemanticSearch dual-collection behavior
  - Dual-indexing in _index_trace_callback
  - UnifiedSearch code collection mapping
  - Config code embedding fields

All tests mock the sentence-transformers model to avoid downloading
the ~500MB jina-embeddings-v2-base-code model during tests.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from engram.search.code_embeddings import (
    CodeEmbedder,
    is_code_content,
    build_code_embedding_func,
    CODE_TRACE_KINDS,
    DEFAULT_CODE_MODEL,
    _CODE_PATTERNS,
)


# ======================================================================
# is_code_content tests
# ======================================================================


class TestIsCodeContent:
    """Test the content classifier."""

    def test_code_trace_kinds(self):
        """All CODE_TRACE_KINDS should classify as code."""
        for kind in CODE_TRACE_KINDS:
            assert is_code_content(trace_kind=kind) is True

    def test_non_code_trace_kinds(self):
        """Non-code trace kinds with empty content should not classify as code."""
        for kind in ["episode", "reflection", "factual", "summary", "thread", "arc"]:
            assert is_code_content(trace_kind=kind, content="") is False

    def test_python_code_content(self):
        """Python code should be detected via regex."""
        samples = [
            "def hello_world():\n    print('hello')",
            "class MyService:\n    pass",
            "import os\nimport sys",
            "from pathlib import Path",
            "async def fetch_data():\n    pass",
        ]
        for sample in samples:
            assert is_code_content(content=sample) is True, f"Failed: {sample[:30]}"

    def test_javascript_code_content(self):
        """JavaScript code should be detected via regex."""
        samples = [
            "function greet(name) { return 'hello'; }",
            "const app = express();",
            "let count = 0;",
            "export default MyComponent;",
            "import React from 'react';",
        ]
        for sample in samples:
            assert is_code_content(content=sample) is True, f"Failed: {sample[:30]}"

    def test_typescript_code_content(self):
        """TypeScript-specific patterns should be detected."""
        samples = [
            "interface UserProps {\n  name: string;\n}",
            "type Status = 'active' | 'inactive';",
            "enum Direction {\n  Up, Down\n}",
        ]
        for sample in samples:
            assert is_code_content(content=sample) is True, f"Failed: {sample[:30]}"

    def test_stack_traces(self):
        """Stack traces should be classified as code."""
        samples = [
            "Traceback (most recent call last):\n  File 'app.py', line 5",
            "Error: Cannot find module 'express'",
            "TypeError: undefined is not a function\n  at processQueue (internal/queue.js:32)",
        ]
        for sample in samples:
            assert is_code_content(content=sample) is True, f"Failed: {sample[:30]}"

    def test_fenced_code_blocks(self):
        """Fenced code blocks with language tags should be detected."""
        samples = [
            "Here's the fix:\n```python\ndef fix(): pass\n```",
            "Use this:\n```javascript\nconst x = 1;\n```",
            "```typescript\ninterface Foo {}\n```",
        ]
        for sample in samples:
            assert is_code_content(content=sample) is True, f"Failed: {sample[:30]}"

    def test_dot_chains(self):
        """Dot chains like a.b.c should be detected."""
        assert is_code_content(content="Call request.auth.user to get the user") is True

    def test_arrow_functions(self):
        """Arrow function syntax should be detected."""
        assert (
            is_code_content(content="const handler = (req) => { return res; }") is True
        )

    def test_plain_text_not_code(self):
        """Plain natural language should not be classified as code."""
        samples = [
            "The meeting went well today.",
            "We decided to use PostgreSQL for the database.",
            "Alice mentioned she prefers dark themes.",
            "Fixed the issue with the login page.",
            "Need to review the architecture tomorrow.",
        ]
        for sample in samples:
            assert is_code_content(content=sample) is False, (
                f"False positive: {sample[:30]}"
            )

    def test_trace_kind_overrides_content(self):
        """Trace kind classification takes priority over content."""
        # Code kind with NL content → still code
        assert is_code_content(trace_kind="code_symbols", content="just text") is True
        # NL kind with code content → detected via content
        assert is_code_content(trace_kind="episode", content="def foo(): pass") is True
        # NL kind with NL content → not code
        assert is_code_content(trace_kind="episode", content="plain text") is False

    def test_empty_inputs(self):
        """Empty inputs should not classify as code."""
        assert is_code_content() is False
        assert is_code_content(trace_kind=None, content="") is False

    def test_code_trace_kinds_complete(self):
        """All expected code trace kinds are present."""
        expected = {
            "code_symbols",
            "code_pattern",
            "debug_session",
            "error_resolution",
            "test_strategy",
            "code_review",
            "wiring_map",
        }
        assert CODE_TRACE_KINDS == expected


# ======================================================================
# CodeEmbedder tests
# ======================================================================


class TestCodeEmbedder:
    """Test the CodeEmbedder class."""

    def test_default_model_name(self):
        """Default model should be jina-embeddings-v2-base-code."""
        embedder = CodeEmbedder()
        assert embedder.model_name == DEFAULT_CODE_MODEL

    def test_custom_model_name(self):
        """Custom model name should be respected."""
        embedder = CodeEmbedder(model_name="custom/model")
        assert embedder.model_name == "custom/model"

    def test_available_when_sentence_transformers_present(self):
        """Should report available when sentence-transformers is importable."""
        embedder = CodeEmbedder()
        # sentence-transformers IS installed in this env
        assert embedder.available is True

    def test_available_cached(self):
        """Availability check should be cached after first call."""
        embedder = CodeEmbedder()
        _ = embedder.available
        # Second call should use cache
        assert embedder._available is not None
        result = embedder.available
        assert result == embedder._available

    @patch(
        "engram.search.code_embeddings.CodeEmbedder.available",
        new_callable=PropertyMock,
        return_value=False,
    )
    def test_encode_returns_none_when_unavailable(self, mock_avail):
        """encode() should return None when model is unavailable."""
        embedder = CodeEmbedder()
        result = embedder.encode(["def foo(): pass"])
        assert result is None

    @patch(
        "engram.search.code_embeddings.CodeEmbedder.available",
        new_callable=PropertyMock,
        return_value=False,
    )
    def test_encode_single_returns_none_when_unavailable(self, mock_avail):
        """encode_single() should return None when model is unavailable."""
        embedder = CodeEmbedder()
        result = embedder.encode_single("def foo(): pass")
        assert result is None

    def test_encode_empty_list(self):
        """Encoding empty list should return empty list."""
        embedder = CodeEmbedder()
        result = embedder.encode([])
        assert result == []

    def test_dimension_none_before_load(self):
        """Dimension should be None before model is loaded."""
        embedder = CodeEmbedder()
        assert embedder.dimension is None

    @patch("engram.search.code_embeddings.CodeEmbedder._load_model")
    def test_encode_calls_load_model(self, mock_load):
        """encode() should attempt to load model."""
        embedder = CodeEmbedder()
        embedder._available = True
        embedder._model = None
        # _load_model is mocked, so _model stays None → returns None
        result = embedder.encode(["test"])
        mock_load.assert_called_once()

    def test_encode_with_mock_model(self):
        """encode() should work with a mock model."""
        import numpy as np

        embedder = CodeEmbedder()
        embedder._available = True

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        embedder._model = mock_model

        result = embedder.encode(["text1", "text2"])
        assert result is not None
        assert len(result) == 2
        assert len(result[0]) == 3
        assert result[0] == pytest.approx([0.1, 0.2, 0.3])

    def test_encode_single_with_mock_model(self):
        """encode_single() should return a single embedding."""
        import numpy as np

        embedder = CodeEmbedder()
        embedder._available = True

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        embedder._model = mock_model

        result = embedder.encode_single("test text")
        assert result is not None
        assert len(result) == 3
        assert result == pytest.approx([0.1, 0.2, 0.3])


# ======================================================================
# build_code_embedding_func tests
# ======================================================================


class TestBuildCodeEmbeddingFunc:
    """Test the factory function."""

    def test_returns_callable_when_available(self):
        """Should return a callable when sentence-transformers is available."""
        func = build_code_embedding_func()
        assert func is not None
        assert callable(func)

    @patch(
        "engram.search.code_embeddings.CodeEmbedder.available",
        new_callable=PropertyMock,
        return_value=False,
    )
    def test_returns_none_when_unavailable(self, mock_avail):
        """Should return None when sentence-transformers is not available."""
        func = build_code_embedding_func()
        assert func is None

    def test_custom_model_name(self):
        """Should accept custom model name."""
        func = build_code_embedding_func(model_name="custom/model")
        assert func is not None
        assert callable(func)


# ======================================================================
# Config integration tests
# ======================================================================


class TestConfigCodeEmbedding:
    """Test Config code embedding fields."""

    def test_default_code_embedding_model(self):
        from engram.core.config import Config

        cfg = Config()
        assert cfg.code_embedding_model == "jinaai/jina-embeddings-v2-base-code"

    def test_default_code_embedding_device(self):
        from engram.core.config import Config

        cfg = Config()
        assert cfg.code_embedding_device == ""

    def test_disabled_code_embedding(self):
        from engram.core.config import Config

        cfg = Config(code_embedding_model="")
        func = cfg.get_code_embedding_func()
        assert func is None

    def test_to_dict_includes_code_embedding(self):
        from engram.core.config import Config

        cfg = Config()
        d = cfg.to_dict()
        assert "code_embedding_model" in d
        assert "code_embedding_device" in d
        assert d["code_embedding_model"] == "jinaai/jina-embeddings-v2-base-code"


# ======================================================================
# SemanticSearch dual-collection tests
# ======================================================================


class TestSemanticSearchDualCollection:
    """Test SemanticSearch with dual-embedding support."""

    def _make_mock_embed(self, dim=3):
        """Create a mock embedding function."""
        import random

        def embed(text):
            random.seed(hash(text) % (2**31))
            return [random.random() for _ in range(dim)]

        return embed

    def test_no_code_collection_without_func(self, tmp_path):
        """Without code_embedding_func, no code collection should exist."""
        from engram.search.semantic import SemanticSearch

        ss = SemanticSearch(
            embeddings_dir=tmp_path / "emb",
            embedding_func=self._make_mock_embed(),
        )
        assert ss.has_code_embeddings is False
        assert "code" not in ss.collection_names
        ss.close()

    def test_code_collection_with_func(self, tmp_path):
        """With code_embedding_func, code collection should be created."""
        from engram.search.semantic import SemanticSearch

        ss = SemanticSearch(
            embeddings_dir=tmp_path / "emb",
            embedding_func=self._make_mock_embed(),
            code_embedding_func=self._make_mock_embed(),
        )
        assert ss.has_code_embeddings is True
        assert "code" in ss.collection_names
        ss.close()

    def test_nl_collections_always_present(self, tmp_path):
        """NL collections should always be present."""
        from engram.search.semantic import SemanticSearch, NL_COLLECTIONS

        ss = SemanticSearch(
            embeddings_dir=tmp_path / "emb",
            embedding_func=self._make_mock_embed(),
            code_embedding_func=self._make_mock_embed(),
        )
        for name in NL_COLLECTIONS:
            assert name in ss.collection_names
        ss.close()

    def test_index_code_succeeds(self, tmp_path):
        """index_code() should succeed when code collection exists."""
        from engram.search.semantic import SemanticSearch

        ss = SemanticSearch(
            embeddings_dir=tmp_path / "emb",
            embedding_func=self._make_mock_embed(),
            code_embedding_func=self._make_mock_embed(),
        )
        result = ss.index_code(
            doc_id="code_abc123",
            content="def hello(): pass",
            metadata={"trace_id": "abc123", "kind": "code_symbols"},
        )
        assert result is True
        ss.close()

    def test_index_code_fails_without_collection(self, tmp_path):
        """index_code() should return False without code collection."""
        from engram.search.semantic import SemanticSearch

        ss = SemanticSearch(
            embeddings_dir=tmp_path / "emb",
            embedding_func=self._make_mock_embed(),
        )
        result = ss.index_code(
            doc_id="code_abc123",
            content="def hello(): pass",
            metadata={"trace_id": "abc123"},
        )
        assert result is False
        ss.close()

    def test_search_includes_code_collection(self, tmp_path):
        """Search with no collection filter should include code collection."""
        from engram.search.semantic import SemanticSearch

        ss = SemanticSearch(
            embeddings_dir=tmp_path / "emb",
            embedding_func=self._make_mock_embed(),
            code_embedding_func=self._make_mock_embed(),
        )
        # Index something in code collection
        ss.index_code(
            doc_id="code_test1",
            content="def verify_token(jwt_string): decode and validate JWT",
            metadata={"trace_id": "test1", "kind": "code_symbols"},
        )
        # Search should find it
        results = ss.search("verify_token JWT")
        found_code = any(r["collection"] == "code" for r in results)
        assert found_code is True
        ss.close()

    def test_dual_indexing(self, tmp_path):
        """Same content can be indexed in both episodic and code collections."""
        from engram.search.semantic import SemanticSearch

        ss = SemanticSearch(
            embeddings_dir=tmp_path / "emb",
            embedding_func=self._make_mock_embed(),
            code_embedding_func=self._make_mock_embed(),
        )
        content = "def authenticate(user, password): validate credentials"
        meta = {"kind": "code_symbols", "trace_id": "dual1"}

        # Index in episodic (NL)
        ss.index_trace("dual1", content, meta)
        # Index in code
        ss.index_code("code_dual1", content, meta)

        # Search should find results from both collections
        results = ss.search("authenticate credentials")
        collections_found = {r["collection"] for r in results}
        assert "episodic" in collections_found
        assert "code" in collections_found
        ss.close()

    def test_reindex_dual_indexes_code_traces(self, tmp_path):
        """reindex_all should dual-index code traces."""
        from engram.search.semantic import SemanticSearch
        from engram.episodic.store import EpisodicStore

        db_path = tmp_path / "test.db"
        store = EpisodicStore(db_path)
        store.log_trace(
            content="def parse_config(): read YAML config file",
            kind="code_symbols",
            tags=["python"],
            salience=0.8,
        )
        store.log_trace(
            content="Had a good meeting with the team today.",
            kind="episode",
            tags=[],
            salience=0.5,
        )

        ss = SemanticSearch(
            embeddings_dir=tmp_path / "emb",
            embedding_func=self._make_mock_embed(),
            code_embedding_func=self._make_mock_embed(),
        )
        counts = ss.reindex_all(
            episodic_store=store,
            semantic_store={},
            procedural_store=MagicMock(
                list_skills=lambda: [], get_skill=lambda n: None
            ),
        )
        assert counts["episodic"] == 2  # both traces in episodic
        assert counts.get("code", 0) >= 1  # code trace in code collection
        store.close()
        ss.close()

    def test_close_clears_code_collection(self, tmp_path):
        """close() should clear code collection flag."""
        from engram.search.semantic import SemanticSearch

        ss = SemanticSearch(
            embeddings_dir=tmp_path / "emb",
            embedding_func=self._make_mock_embed(),
            code_embedding_func=self._make_mock_embed(),
        )
        assert ss.has_code_embeddings is True
        ss.close()
        assert ss.has_code_embeddings is False


# ======================================================================
# UnifiedSearch code collection mapping tests
# ======================================================================


class TestUnifiedSearchCodeMapping:
    """Test that UnifiedSearch includes code collection correctly."""

    def test_traces_includes_code(self):
        from engram.search.unified import UnifiedSearch

        collections = UnifiedSearch._map_collections("traces")
        assert "episodic" in collections
        assert "code" in collections

    def test_messages_excludes_code(self):
        from engram.search.unified import UnifiedSearch

        collections = UnifiedSearch._map_collections("messages")
        assert collections == ["episodic"]
        assert "code" not in collections

    def test_none_returns_none(self):
        """None memory_type should return None (search all)."""
        from engram.search.unified import UnifiedSearch

        collections = UnifiedSearch._map_collections(None)
        assert collections is None


# ======================================================================
# MemoryStats code_embeddings_active tests
# ======================================================================


class TestMemoryStatsCodeEmbeddings:
    """Test the code_embeddings_active field in MemoryStats."""

    def test_default_false(self):
        from engram.core.types import MemoryStats

        stats = MemoryStats()
        assert stats.code_embeddings_active is False

    def test_to_dict_includes_field(self):
        from engram.core.types import MemoryStats

        stats = MemoryStats(code_embeddings_active=True)
        d = stats.to_dict()
        assert "code_embeddings_active" in d
        assert d["code_embeddings_active"] is True

    def test_set_true(self):
        from engram.core.types import MemoryStats

        stats = MemoryStats(code_embeddings_active=True)
        assert stats.code_embeddings_active is True


# ======================================================================
# Integration test: full pipeline with mock embeddings
# ======================================================================


class TestDualEmbeddingIntegration:
    """Integration test: MemorySystem with dual embeddings."""

    def _make_mock_embed(self, dim=3):
        import random

        def embed(text):
            random.seed(hash(text) % (2**31))
            return [random.random() for _ in range(dim)]

        return embed

    def test_system_with_code_embedding(self, tmp_path):
        """MemorySystem should wire code embeddings when configured."""
        from engram.core.config import Config
        from engram.system import MemorySystem

        cfg = Config.from_data_dir(
            tmp_path,
            signal_mode="regex",
            extract_mode="off",
            code_embedding_model="",  # Disable to avoid model download
        )
        cfg.ensure_directories()

        system = MemorySystem(
            config=cfg,
            embedding_func=self._make_mock_embed(),
        )
        # Without code embedding model, should not have code embeddings
        _ = system.unified_search  # force lazy init
        assert system._semantic_search is not None
        assert system._semantic_search.has_code_embeddings is False
        system.close()

    def test_system_with_code_embedding_func(self, tmp_path):
        """MemorySystem with explicit code embedding func should activate dual-embedding."""
        from engram.core.config import Config
        from engram.system import MemorySystem

        cfg = Config.from_data_dir(
            tmp_path,
            signal_mode="regex",
            extract_mode="off",
            code_embedding_model="",  # Don't try to build from config
        )
        cfg.ensure_directories()

        # Manually inject code embedding func
        system = MemorySystem(
            config=cfg,
            embedding_func=self._make_mock_embed(),
        )
        system._code_embedding_func = self._make_mock_embed()
        # Force re-init of semantic search
        system._unified_search = None
        system._semantic_search = None
        _ = system.unified_search

        assert system._semantic_search is not None
        assert system._semantic_search.has_code_embeddings is True
        system.close()

    def test_trace_callback_dual_indexes(self, tmp_path):
        """_index_trace_callback should dual-index code traces."""
        from engram.core.config import Config
        from engram.system import MemorySystem

        cfg = Config.from_data_dir(
            tmp_path,
            signal_mode="regex",
            extract_mode="off",
            code_embedding_model="",
        )
        cfg.ensure_directories()

        system = MemorySystem(
            config=cfg,
            embedding_func=self._make_mock_embed(),
        )
        system._code_embedding_func = self._make_mock_embed()
        system._unified_search = None
        system._semantic_search = None
        _ = system.unified_search

        assert system._semantic_search.has_code_embeddings is True

        # Log a code trace — should trigger dual-indexing via callback
        trace_id = system.episodic.log_trace(
            content="def parse_config(): read config file",
            kind="code_symbols",
            tags=["python"],
            salience=0.8,
        )

        # Check code collection has the document
        code_col = system._semantic_search._collections.get("code")
        assert code_col is not None
        assert code_col.count() >= 1

        system.close()
