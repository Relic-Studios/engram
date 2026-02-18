"""Tests for engram.extraction.symbol_index — Symbol indexing and repo-map."""

import json
import os
import tempfile

import pytest

from engram.extraction.ast_engine import SymbolKind
from engram.extraction.symbol_index import FileSymbols, SymbolIndex


# ---------------------------------------------------------------------------
# FileSymbols
# ---------------------------------------------------------------------------


class TestFileSymbols:
    def test_round_trip(self):
        index = SymbolIndex()
        code = "def foo(): pass\ndef bar(): pass"
        fs = index.index_file("test.py", code)
        d = fs.to_dict()
        fs2 = FileSymbols.from_dict(d)
        assert fs2.file_path == "test.py"
        assert len(fs2.symbols) == len(fs.symbols)

    def test_exported_names(self):
        index = SymbolIndex()
        code = "def public(): pass\ndef _private(): pass"
        fs = index.index_file("test.py", code)
        exported = fs.exported_names
        assert "public" in exported
        assert "_private" not in exported

    def test_import_modules(self):
        index = SymbolIndex()
        code = "import os\nfrom typing import List"
        fs = index.index_file("test.py", code)
        assert "os" in fs.import_modules
        assert "typing" in fs.import_modules

    def test_repo_map_entry(self):
        index = SymbolIndex()
        code = """
class MyClass:
    def method(self): pass

def helper(): pass
"""
        fs = index.index_file("mymodule.py", code)
        entry = fs.to_repo_map_entry()
        assert "mymodule.py:" in entry
        assert "class MyClass" in entry
        assert "def helper" in entry


# ---------------------------------------------------------------------------
# SymbolIndex — basic operations
# ---------------------------------------------------------------------------


class TestSymbolIndex:
    def test_index_file(self):
        index = SymbolIndex()
        code = "def foo(): pass"
        fs = index.index_file("test.py", code)
        assert index.file_count == 1
        assert index.symbol_count >= 1

    def test_index_multiple_files(self):
        index = SymbolIndex()
        index.index_file("a.py", "def foo(): pass")
        index.index_file("b.py", "def bar(): pass")
        assert index.file_count == 2

    def test_incremental_indexing(self):
        """Re-indexing same content should be a no-op."""
        index = SymbolIndex()
        code = "def foo(): pass"
        fs1 = index.index_file("test.py", code)
        fs2 = index.index_file("test.py", code)
        assert fs1.fingerprint == fs2.fingerprint

    def test_update_on_content_change(self):
        index = SymbolIndex()
        index.index_file("test.py", "def foo(): pass")
        index.index_file("test.py", "def bar(): pass")
        # Should have updated, not duplicated
        assert index.file_count == 1
        results = index.find_symbol("bar")
        assert len(results) == 1
        # Old symbol should be gone
        assert len(index.find_symbol("foo")) == 0

    def test_remove_file(self):
        index = SymbolIndex()
        index.index_file("test.py", "def foo(): pass")
        assert index.file_count == 1
        removed = index.remove_file("test.py")
        assert removed
        assert index.file_count == 0
        assert len(index.find_symbol("foo")) == 0

    def test_remove_nonexistent(self):
        index = SymbolIndex()
        assert not index.remove_file("nonexistent.py")


# ---------------------------------------------------------------------------
# SymbolIndex — lookup
# ---------------------------------------------------------------------------


class TestSymbolLookup:
    @pytest.fixture
    def populated_index(self):
        index = SymbolIndex()
        index.index_file(
            "services/user.py",
            """
class UserService:
    def get_user(self, id: int) -> dict:
        pass
    def create_user(self, name: str) -> dict:
        pass
""",
        )
        index.index_file(
            "services/auth.py",
            """
from services.user import UserService

class AuthService:
    def authenticate(self, token: str) -> bool:
        pass
""",
        )
        index.index_file(
            "app.py",
            """
from services.user import UserService
from services.auth import AuthService

def create_app():
    return App()
""",
        )
        return index

    def test_find_symbol(self, populated_index):
        results = populated_index.find_symbol("UserService")
        assert len(results) == 1
        assert results[0]["file_path"] == "services/user.py"

    def test_find_symbol_not_found(self, populated_index):
        results = populated_index.find_symbol("NonExistent")
        assert len(results) == 0

    def test_find_symbols_by_kind(self, populated_index):
        classes = populated_index.find_symbols_by_kind(SymbolKind.CLASS)
        names = {c["symbol"]["name"] for c in classes}
        assert "UserService" in names
        assert "AuthService" in names

    def test_get_file_symbols(self, populated_index):
        fs = populated_index.get_file_symbols("services/user.py")
        assert fs is not None
        assert fs.language == "python"
        names = {s.name for s in fs.symbols}
        assert "UserService" in names

    def test_get_imports_for(self, populated_index):
        imports = populated_index.get_imports_for("app.py")
        modules = {i.module for i in imports}
        assert "services.user" in modules
        assert "services.auth" in modules

    def test_get_dependents(self, populated_index):
        deps = populated_index.get_dependents("services.user")
        assert "app.py" in deps

    def test_get_dependency_edges(self, populated_index):
        edges = populated_index.get_dependency_edges()
        # Should have depends_on and exports edges
        predicates = {e["predicate"] for e in edges}
        assert "depends_on" in predicates
        assert "exports" in predicates

        # Check specific edges
        dep_edges = [e for e in edges if e["predicate"] == "depends_on"]
        sources = {e["source"] for e in dep_edges}
        assert "app.py" in sources


# ---------------------------------------------------------------------------
# Repo-map generation
# ---------------------------------------------------------------------------


class TestRepoMap:
    def test_basic_map(self):
        index = SymbolIndex()
        index.index_file(
            "lib.py",
            """
class Engine:
    def start(self): pass
    def stop(self): pass

def create_engine() -> Engine:
    return Engine()
""",
        )
        repo_map = index.generate_repo_map()
        assert "lib.py:" in repo_map
        assert "class Engine" in repo_map
        assert "def create_engine" in repo_map

    def test_map_with_budget(self):
        index = SymbolIndex()
        for i in range(20):
            code = f"def func_{i}(): pass\ndef other_{i}(): pass"
            index.index_file(f"module_{i}.py", code)

        # Very small budget should truncate
        small_map = index.generate_repo_map(max_tokens=100)
        large_map = index.generate_repo_map(max_tokens=10000)
        assert len(small_map) < len(large_map)

    def test_map_with_file_filter(self):
        index = SymbolIndex()
        index.index_file("a.py", "def aa(): pass")
        index.index_file("b.py", "def bb(): pass")

        filtered = index.generate_repo_map(file_filter=["a.py"])
        assert "a.py:" in filtered
        assert "b.py:" not in filtered


# ---------------------------------------------------------------------------
# Directory indexing
# ---------------------------------------------------------------------------


class TestDirectoryIndexing:
    def test_index_directory(self, tmp_path):
        # Create a small project
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.py").write_text("class Inner: pass")

        index = SymbolIndex()
        count = index.index_directory(str(tmp_path))
        assert count == 3
        assert index.file_count == 3

    def test_exclude_patterns(self, tmp_path):
        (tmp_path / "main.py").write_text("def main(): pass")
        excluded = tmp_path / "__pycache__"
        excluded.mkdir()
        (excluded / "cached.py").write_text("def cached(): pass")

        index = SymbolIndex()
        count = index.index_directory(str(tmp_path))
        assert count == 1  # Only main.py, not __pycache__/cached.py

    def test_extension_filter(self, tmp_path):
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "style.css").write_text("body { color: red; }")
        (tmp_path / "data.json").write_text('{"key": "value"}')

        index = SymbolIndex()
        count = index.index_directory(str(tmp_path), extensions={".py"})
        assert count == 1


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_json_from_json(self):
        index = SymbolIndex()
        index.index_file("a.py", "def foo(): pass")
        index.index_file("b.py", "class Bar: pass")

        json_str = index.to_json()
        index2 = SymbolIndex.from_json(json_str)

        assert index2.file_count == 2
        assert index2.symbol_count == index.symbol_count

        # Lookup should work on deserialized index
        results = index2.find_symbol("foo")
        assert len(results) == 1

    def test_to_dict_from_dict(self):
        index = SymbolIndex()
        index.index_file("test.py", "import os\ndef hello(): pass")

        d = index.to_dict()
        index2 = SymbolIndex.from_dict(d)

        assert index2.file_count == 1
        fs = index2.get_file_symbols("test.py")
        assert fs is not None
        assert len(fs.imports) == 1
