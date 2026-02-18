"""Tests for engram.search.tokenizer — Code-aware symbol tokenizer."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from engram.search.tokenizer import (
    expand_query,
    expand_text,
    expand_token,
    is_compound_identifier,
    split_identifier,
)


# ---------------------------------------------------------------------------
# split_identifier
# ---------------------------------------------------------------------------


class TestSplitIdentifier:
    def test_camel_case(self):
        assert split_identifier("getUserById") == ["get", "user", "by", "id"]

    def test_pascal_case(self):
        assert split_identifier("UserService") == ["user", "service"]

    def test_snake_case(self):
        assert split_identifier("snake_case_name") == ["snake", "case", "name"]

    def test_upper_snake(self):
        assert split_identifier("MAX_RETRY_COUNT") == ["max", "retry", "count"]

    def test_dot_path(self):
        assert split_identifier("os.path.join") == ["os", "path", "join"]

    def test_acronym(self):
        assert split_identifier("HTTPResponse") == ["http", "response"]

    def test_xml_http_request(self):
        assert split_identifier("XMLHttpRequest") == ["xml", "http", "request"]

    def test_kebab_case(self):
        assert split_identifier("my-kebab-case") == ["my", "kebab", "case"]

    def test_mixed_camel_snake(self):
        assert split_identifier("get_userById") == ["get", "user", "by", "id"]

    def test_dotted_module(self):
        assert split_identifier("engram.signal.measure") == [
            "engram",
            "signal",
            "measure",
        ]

    def test_single_word(self):
        assert split_identifier("simple") == ["simple"]

    def test_short_identifier(self):
        assert split_identifier("x") == []

    def test_empty_string(self):
        assert split_identifier("") == []

    def test_all_caps_single(self):
        # "ID" is too short after split
        result = split_identifier("ID")
        assert result == ["id"]

    def test_preserves_lowercasing(self):
        parts = split_identifier("GetUser")
        assert all(p.islower() for p in parts)


# ---------------------------------------------------------------------------
# is_compound_identifier
# ---------------------------------------------------------------------------


class TestIsCompound:
    def test_camel_case(self):
        assert is_compound_identifier("getUserById")

    def test_snake_case(self):
        assert is_compound_identifier("snake_case")

    def test_dot_path(self):
        assert is_compound_identifier("os.path")

    def test_simple_word(self):
        assert not is_compound_identifier("simple")

    def test_short(self):
        assert not is_compound_identifier("xy")

    def test_empty(self):
        assert not is_compound_identifier("")


# ---------------------------------------------------------------------------
# expand_token
# ---------------------------------------------------------------------------


class TestExpandToken:
    def test_compound(self):
        result = expand_token("getUserById")
        assert "getUserById" in result
        assert "get" in result
        assert "user" in result

    def test_simple(self):
        assert expand_token("simple") == "simple"

    def test_empty(self):
        assert expand_token("") == ""


# ---------------------------------------------------------------------------
# expand_text
# ---------------------------------------------------------------------------


class TestExpandText:
    def test_with_identifiers(self):
        text = "The getUserById function returns HTTPResponse"
        expanded = expand_text(text)
        assert "getUserById" in expanded
        assert "get" in expanded
        assert "user" in expanded
        assert "http" in expanded
        assert "response" in expanded

    def test_no_identifiers(self):
        text = "just plain english text"
        assert expand_text(text) == text

    def test_empty(self):
        assert expand_text("") == ""

    def test_deduplication(self):
        text = "getUserById getUserById getUserById"
        expanded = expand_text(text)
        # Should only expand once, not three times
        parts = expanded.split()
        get_count = parts.count("get")
        assert get_count == 1

    def test_snake_case_in_text(self):
        text = "Use the snake_case_name variable"
        expanded = expand_text(text)
        assert "snake" in expanded
        assert "case" in expanded
        assert "name" in expanded


# ---------------------------------------------------------------------------
# expand_query
# ---------------------------------------------------------------------------


class TestExpandQuery:
    def test_compound_query(self):
        result = expand_query("getUserById error")
        assert "getUserById" in result
        assert "get" in result
        assert "user" in result
        assert "error" in result

    def test_simple_query(self):
        assert expand_query("simple query") == "simple query"

    def test_empty(self):
        assert expand_query("") == ""

    def test_multiple_compounds(self):
        result = expand_query("getUserById snake_case")
        assert "get" in result
        assert "snake" in result
        assert "case" in result

    def test_dot_path_query(self):
        result = expand_query("os.path.join")
        assert "os" in result
        assert "path" in result
        assert "join" in result


# ---------------------------------------------------------------------------
# FTS integration — verify expanded content is searchable
# ---------------------------------------------------------------------------


class TestFTSIntegration:
    """Test that symbol expansion actually improves FTS5 search results."""

    @pytest.fixture
    def store(self, tmp_path):
        from engram.episodic.store import EpisodicStore

        return EpisodicStore(tmp_path / "test.db")

    def test_search_finds_camel_case_parts(self, store):
        """Searching 'user' should find a trace containing 'getUserById'."""
        store.log_trace(
            content="Use the getUserById function to fetch user data",
            kind="episode",
            tags=["test"],
        )
        # Search for a part of the compound identifier
        results = store.search_traces("user")
        assert len(results) >= 1

    def test_search_finds_snake_case_parts(self, store):
        """Searching 'retry' should find content with 'max_retry_count'."""
        store.log_trace(
            content="Set max_retry_count to 5 for resilience",
            kind="episode",
            tags=["test"],
        )
        results = store.search_traces("retry")
        assert len(results) >= 1

    def test_search_finds_dot_path_parts(self, store):
        """Searching 'measure' should find content with 'engram.signal.measure'."""
        store.log_trace(
            content="The engram.signal.measure module handles CQS",
            kind="episode",
            tags=["test"],
        )
        results = store.search_traces("measure")
        assert len(results) >= 1

    def test_search_still_finds_exact_match(self, store):
        """Exact compound identifier search still works."""
        store.log_trace(
            content="The getUserById function is important",
            kind="episode",
            tags=["test"],
        )
        results = store.search_traces("getUserById")
        assert len(results) >= 1

    def test_message_fts_expansion(self, store):
        """Messages also get symbol-expanded FTS content."""
        store.log_message(
            person="test",
            speaker="test",
            content="Call processData to handle the HTTPResponse",
            source="test",
        )
        results = store.search_traces("process")
        # Messages are in messages_fts, not traces_fts
        # But search_traces only searches traces
        # So let's verify via direct FTS query
        rows = store.conn.execute(
            "SELECT * FROM messages_fts WHERE messages_fts MATCH ?",
            ('"process"',),
        ).fetchall()
        assert len(rows) >= 1

    def test_indexed_search_query_expansion(self):
        """IndexedSearch._sanitise_query expands compound identifiers."""
        from engram.search.indexed import IndexedSearch

        result = IndexedSearch._sanitise_query("getUserById error")
        # Should contain expanded parts
        assert '"get"' in result
        assert '"user"' in result
        assert '"error"' in result
        # Original should also be present
        assert '"getUserById"' in result
