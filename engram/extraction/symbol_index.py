"""
engram.extraction.symbol_index — Persistent symbol index for repo-map generation.

Maintains an indexed collection of code symbols extracted from AST analysis,
enabling fast lookup by name, kind, file, and dependency relationships.
Generates Aider-style repo-maps for context injection.

Storage uses the existing episodic store (traces table with kind='code_symbols')
and relationships table for wiring edges.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from engram.extraction.ast_engine import (
    CodeAnalysis,
    ImportInfo,
    Symbol,
    SymbolKind,
    analyze_code,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File-level symbol record
# ---------------------------------------------------------------------------


@dataclass
class FileSymbols:
    """Symbols extracted from a single file."""

    file_path: str
    language: str
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    fingerprint: str = ""  # content hash for change detection

    @property
    def exported_names(self) -> List[str]:
        return [s.name for s in self.symbols if s.is_exported]

    @property
    def import_modules(self) -> List[str]:
        return [i.module for i in self.imports]

    def to_dict(self) -> Dict:
        return {
            "file_path": self.file_path,
            "language": self.language,
            "symbols": [s.to_dict() for s in self.symbols],
            "imports": [i.to_dict() for i in self.imports],
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "FileSymbols":
        symbols = [
            Symbol(
                name=s["name"],
                kind=SymbolKind(s["kind"]),
                line=s.get("line", 0),
                end_line=s.get("end_line", 0),
                signature=s.get("signature", ""),
                docstring=s.get("docstring", ""),
                parent=s.get("parent", ""),
                decorators=s.get("decorators", []),
                is_exported=s.get("is_exported", True),
                type_annotation=s.get("type_annotation", ""),
            )
            for s in d.get("symbols", [])
        ]
        imports = [
            ImportInfo(
                module=i["module"],
                names=i.get("names", []),
                alias=i.get("alias", ""),
                is_relative=i.get("is_relative", False),
                line=i.get("line", 0),
                is_wildcard=i.get("is_wildcard", False),
            )
            for i in d.get("imports", [])
        ]
        return cls(
            file_path=d["file_path"],
            language=d.get("language", "unknown"),
            symbols=symbols,
            imports=imports,
            fingerprint=d.get("fingerprint", ""),
        )

    def to_repo_map_entry(self) -> str:
        """Format as a repo-map entry for context injection."""
        lines: List[str] = [self.file_path + ":"]
        for s in self.symbols:
            indent = "    " if s.parent else "  "
            if s.kind == SymbolKind.CLASS:
                lines.append(f"{indent}class {s.name}")
            elif s.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                sig = s.signature or f"def {s.name}(...)"
                lines.append(f"{indent}{sig}")
            elif s.kind == SymbolKind.INTERFACE:
                lines.append(f"{indent}interface {s.name}")
            elif s.kind == SymbolKind.TYPE_ALIAS:
                lines.append(f"{indent}type {s.name}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Symbol Index
# ---------------------------------------------------------------------------


class SymbolIndex:
    """
    In-memory symbol index with persistence via episodic store.

    Maintains a mapping of file paths to their extracted symbols,
    enabling:
      - Fast symbol lookup by name (across files)
      - Dependency graph construction from imports
      - Repo-map generation for context injection
      - Change detection via content fingerprinting

    Usage::

        index = SymbolIndex()
        index.index_file("/path/to/file.py", code)
        map_text = index.generate_repo_map()
    """

    def __init__(self) -> None:
        self._files: Dict[str, FileSymbols] = {}
        self._symbol_lookup: Dict[str, List[str]] = {}  # name -> [file_paths]

    @property
    def file_count(self) -> int:
        return len(self._files)

    @property
    def symbol_count(self) -> int:
        return sum(len(fs.symbols) for fs in self._files.values())

    def index_file(self, file_path: str, code: str, language: str = "") -> FileSymbols:
        """
        Index a file's symbols from source code.

        If the file was already indexed and the fingerprint matches,
        skips re-analysis (incremental indexing).

        Parameters
        ----------
        file_path : str
            Path to the file (used as key).
        code : str
            Source code content.
        language : str
            Language hint (auto-detected if empty).

        Returns
        -------
        FileSymbols
            The extracted symbols for this file.
        """
        analysis = analyze_code(code, language)

        # Check if we already have this file with same content
        existing = self._files.get(file_path)
        if existing and existing.fingerprint == analysis.fingerprint:
            return existing

        fs = FileSymbols(
            file_path=file_path,
            language=analysis.language,
            symbols=analysis.symbols,
            imports=analysis.imports,
            fingerprint=analysis.fingerprint,
        )

        # Remove old entries from lookup
        if file_path in self._files:
            self._remove_from_lookup(file_path)

        # Store
        self._files[file_path] = fs

        # Update lookup index
        for symbol in fs.symbols:
            if symbol.name not in self._symbol_lookup:
                self._symbol_lookup[symbol.name] = []
            if file_path not in self._symbol_lookup[symbol.name]:
                self._symbol_lookup[symbol.name].append(file_path)

        return fs

    def index_directory(
        self,
        directory: str,
        extensions: Optional[Set[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> int:
        """
        Index all supported files in a directory.

        Parameters
        ----------
        directory : str
            Root directory to scan.
        extensions : set[str], optional
            File extensions to include (e.g. {".py", ".js"}).
            Defaults to common code extensions.
        exclude_patterns : list[str], optional
            Directory name patterns to exclude (e.g. ["node_modules", "__pycache__"]).

        Returns
        -------
        int
            Number of files indexed.
        """
        if extensions is None:
            extensions = {".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go"}
        if exclude_patterns is None:
            exclude_patterns = [
                "node_modules",
                "__pycache__",
                ".git",
                ".venv",
                "venv",
                "dist",
                "build",
                ".tox",
                ".mypy_cache",
                ".pytest_cache",
                "egg-info",
            ]

        count = 0
        for root, dirs, files in os.walk(directory):
            # Filter excluded directories in-place
            dirs[:] = [d for d in dirs if not any(pat in d for pat in exclude_patterns)]

            for fname in files:
                _, ext = os.path.splitext(fname)
                if ext not in extensions:
                    continue

                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        code = f.read()
                    if len(code) > 100_000:  # skip files > 100KB
                        log.debug(
                            "Skipping large file: %s (%d bytes)", fpath, len(code)
                        )
                        continue
                    self.index_file(fpath, code)
                    count += 1
                except (OSError, IOError) as exc:
                    log.debug("Could not read %s: %s", fpath, exc)

        return count

    def remove_file(self, file_path: str) -> bool:
        """Remove a file from the index."""
        if file_path not in self._files:
            return False
        self._remove_from_lookup(file_path)
        del self._files[file_path]
        return True

    def _remove_from_lookup(self, file_path: str) -> None:
        """Remove all symbol lookup entries for a file."""
        fs = self._files.get(file_path)
        if not fs:
            return
        for symbol in fs.symbols:
            if symbol.name in self._symbol_lookup:
                paths = self._symbol_lookup[symbol.name]
                if file_path in paths:
                    paths.remove(file_path)
                if not paths:
                    del self._symbol_lookup[symbol.name]

    # --- Lookup methods ---

    def find_symbol(self, name: str) -> List[Dict]:
        """
        Find all symbols with the given name across all indexed files.

        Returns list of dicts with 'file_path', 'symbol' keys.
        """
        results: List[Dict] = []
        file_paths = self._symbol_lookup.get(name, [])
        for fp in file_paths:
            fs = self._files.get(fp)
            if fs:
                for s in fs.symbols:
                    if s.name == name:
                        results.append({"file_path": fp, "symbol": s.to_dict()})
        return results

    def find_symbols_by_kind(self, kind: SymbolKind) -> List[Dict]:
        """Find all symbols of a specific kind across all files."""
        results: List[Dict] = []
        for fp, fs in self._files.items():
            for s in fs.symbols:
                if s.kind == kind:
                    results.append({"file_path": fp, "symbol": s.to_dict()})
        return results

    def get_file_symbols(self, file_path: str) -> Optional[FileSymbols]:
        """Get symbols for a specific file."""
        return self._files.get(file_path)

    def get_imports_for(self, file_path: str) -> List[ImportInfo]:
        """Get all imports for a specific file."""
        fs = self._files.get(file_path)
        return fs.imports if fs else []

    def get_dependents(self, module_name: str) -> List[str]:
        """
        Find all files that import a given module.

        Parameters
        ----------
        module_name : str
            Module name to search for (e.g., "os.path", "react").

        Returns
        -------
        list[str]
            File paths that import this module.
        """
        results: List[str] = []
        for fp, fs in self._files.items():
            for imp in fs.imports:
                if imp.module == module_name or imp.module.endswith(f".{module_name}"):
                    results.append(fp)
                    break
        return results

    def get_dependency_edges(self) -> List[Dict]:
        """
        Build dependency edges from import analysis.

        Returns list of dicts with 'source', 'target', 'names' keys,
        suitable for storing as relationships in the episodic store.
        """
        edges: List[Dict] = []
        for fp, fs in self._files.items():
            for imp in fs.imports:
                if imp.module:
                    edges.append(
                        {
                            "source": fp,
                            "target": imp.module,
                            "predicate": "depends_on",
                            "names": imp.names,
                            "is_relative": imp.is_relative,
                            "is_wildcard": imp.is_wildcard,
                        }
                    )
            for s in fs.symbols:
                if s.is_exported and s.kind in (
                    SymbolKind.FUNCTION,
                    SymbolKind.CLASS,
                    SymbolKind.INTERFACE,
                ):
                    edges.append(
                        {
                            "source": fp,
                            "target": s.name,
                            "predicate": "exports",
                        }
                    )
        return edges

    # --- Repo-map generation ---

    def generate_repo_map(
        self,
        max_tokens: int = 2000,
        include_signatures: bool = True,
        file_filter: Optional[List[str]] = None,
    ) -> str:
        """
        Generate an Aider-style repo-map for context injection.

        The repo-map is a compact representation of the repository structure
        showing file paths, class definitions, and function signatures.

        Parameters
        ----------
        max_tokens : int
            Approximate token budget for the map (4 chars ~ 1 token).
        include_signatures : bool
            Include function signatures (vs just names).
        file_filter : list[str], optional
            If provided, only include these file paths.

        Returns
        -------
        str
            Compact repo-map text.
        """
        lines: List[str] = []
        char_budget = max_tokens * 4  # rough token estimate

        # Sort files for consistent output
        file_paths = sorted(self._files.keys())
        if file_filter:
            file_paths = [fp for fp in file_paths if fp in file_filter]

        current_chars = 0
        for fp in file_paths:
            fs = self._files[fp]
            entry = fs.to_repo_map_entry()
            entry_len = len(entry)

            if current_chars + entry_len > char_budget:
                # Budget exceeded — add abbreviated entry
                lines.append(f"{fp}: ({len(fs.symbols)} symbols)")
                current_chars += len(fp) + 20
                if current_chars > char_budget:
                    lines.append(f"... and {len(file_paths) - len(lines)} more files")
                    break
            else:
                lines.append(entry)
                current_chars += entry_len

        return "\n".join(lines)

    # --- Serialization ---

    def to_dict(self) -> Dict:
        """Serialize the entire index to a dict."""
        return {
            "files": {fp: fs.to_dict() for fp, fs in self._files.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SymbolIndex":
        """Deserialize from a dict."""
        index = cls()
        for fp, fs_data in data.get("files", {}).items():
            fs = FileSymbols.from_dict(fs_data)
            index._files[fp] = fs
            for symbol in fs.symbols:
                if symbol.name not in index._symbol_lookup:
                    index._symbol_lookup[symbol.name] = []
                if fp not in index._symbol_lookup[symbol.name]:
                    index._symbol_lookup[symbol.name].append(fp)
        return index

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "SymbolIndex":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
