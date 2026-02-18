"""
engram.extraction.ast_engine — Multi-language AST extraction engine.

Uses tree-sitter for language-agnostic parsing (Python, JavaScript,
TypeScript) and Python's stdlib ``ast`` module for deep Python analysis
(cyclomatic complexity, scope-aware naming, type annotation coverage).

Design principles:
  - Zero API dependencies — everything runs locally
  - Graceful degradation — if tree-sitter grammars aren't installed,
    falls back to regex heuristics (already in measure.py)
  - Returns structured dataclasses, not raw AST nodes
  - Sub-second for typical LLM response code blocks (<500 LOC)
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tree-sitter lazy loading
# ---------------------------------------------------------------------------

_TS_LANGUAGES: Dict[str, object] = {}
_TS_AVAILABLE = False

try:
    import tree_sitter as _ts

    _TS_AVAILABLE = True
except ImportError:
    _ts = None  # type: ignore[assignment]


def _get_ts_language(lang: str) -> Optional[object]:
    """Lazily load and cache a tree-sitter Language object."""
    if not _TS_AVAILABLE:
        return None
    if lang in _TS_LANGUAGES:
        return _TS_LANGUAGES[lang]

    try:
        if lang == "python":
            import tree_sitter_python as tsp

            language = _ts.Language(tsp.language())
        elif lang == "javascript":
            import tree_sitter_javascript as tsj

            language = _ts.Language(tsj.language())
        elif lang in ("typescript", "tsx"):
            import tree_sitter_typescript as tst

            ts_lang = tst.language_tsx() if lang == "tsx" else tst.language_typescript()
            language = _ts.Language(ts_lang)
        else:
            return None

        _TS_LANGUAGES[lang] = language
        return language
    except (ImportError, Exception) as exc:
        log.debug("tree-sitter grammar for %s not available: %s", lang, exc)
        return None


def _ts_parse(code: str, lang: str) -> Optional[object]:
    """Parse code with tree-sitter, returning a tree or None."""
    language = _get_ts_language(lang)
    if language is None:
        return None
    try:
        parser = _ts.Parser(language)
        return parser.parse(code.encode("utf-8"))
    except Exception as exc:
        log.debug("tree-sitter parse failed for %s: %s", lang, exc)
        return None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class SymbolKind(str, Enum):
    """Kind of code symbol extracted from AST."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    MODULE = "module"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    DECORATOR = "decorator"
    PROPERTY = "property"


@dataclass
class Symbol:
    """A code symbol extracted from AST analysis."""

    name: str
    kind: SymbolKind
    line: int  # 1-indexed line number
    end_line: int = 0  # end line (0 = same as start)
    signature: str = ""  # e.g. "def foo(x: int, y: str) -> bool"
    docstring: str = ""  # first line of docstring if present
    parent: str = ""  # parent symbol name (for methods/nested)
    decorators: List[str] = field(default_factory=list)
    is_exported: bool = True  # public (no leading underscore)
    type_annotation: str = ""  # return type or variable type

    @property
    def qualified_name(self) -> str:
        """Dot-separated qualified name including parent."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "line": self.line,
            "end_line": self.end_line,
            "signature": self.signature,
            "docstring": self.docstring,
            "parent": self.parent,
            "decorators": self.decorators,
            "is_exported": self.is_exported,
            "type_annotation": self.type_annotation,
            "qualified_name": self.qualified_name,
        }


@dataclass
class ImportInfo:
    """An import statement extracted from code."""

    module: str  # "os.path" or "react"
    names: List[str] = field(default_factory=list)  # ["join", "dirname"]
    alias: str = ""  # "import numpy as np" -> alias = "np"
    is_relative: bool = False  # from . import ...
    line: int = 0
    is_wildcard: bool = False  # from x import *

    def to_dict(self) -> Dict:
        return {
            "module": self.module,
            "names": self.names,
            "alias": self.alias,
            "is_relative": self.is_relative,
            "line": self.line,
            "is_wildcard": self.is_wildcard,
        }


@dataclass
class ComplexityMetrics:
    """Structural complexity metrics from AST analysis."""

    cyclomatic_complexity: int = 1  # McCabe's cyclomatic complexity
    max_nesting_depth: int = 0  # deepest nesting level
    num_functions: int = 0
    num_classes: int = 0
    num_imports: int = 0
    num_lines: int = 0
    num_blank_lines: int = 0
    num_comment_lines: int = 0
    has_type_annotations: bool = False
    type_annotation_coverage: float = 0.0  # 0-1, fraction of funcs with annotations
    has_docstrings: bool = False
    docstring_coverage: float = 0.0  # 0-1, fraction of funcs/classes with docstrings
    has_tests: bool = False  # any function starting with test_
    assertion_density: float = 0.0  # assertions per function
    # Anti-patterns
    bare_except_count: int = 0
    broad_except_count: int = 0
    eval_exec_count: int = 0
    wildcard_import_count: int = 0
    empty_block_count: int = 0  # pass in except/if

    def to_dict(self) -> Dict:
        return {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "max_nesting_depth": self.max_nesting_depth,
            "num_functions": self.num_functions,
            "num_classes": self.num_classes,
            "num_imports": self.num_imports,
            "num_lines": self.num_lines,
            "num_blank_lines": self.num_blank_lines,
            "num_comment_lines": self.num_comment_lines,
            "has_type_annotations": self.has_type_annotations,
            "type_annotation_coverage": round(self.type_annotation_coverage, 4),
            "has_docstrings": self.has_docstrings,
            "docstring_coverage": round(self.docstring_coverage, 4),
            "has_tests": self.has_tests,
            "assertion_density": round(self.assertion_density, 4),
            "bare_except_count": self.bare_except_count,
            "broad_except_count": self.broad_except_count,
            "eval_exec_count": self.eval_exec_count,
            "wildcard_import_count": self.wildcard_import_count,
            "empty_block_count": self.empty_block_count,
        }


@dataclass
class CodeAnalysis:
    """Complete analysis result for a code block."""

    language: str
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    complexity: ComplexityMetrics = field(default_factory=ComplexityMetrics)
    parse_errors: List[str] = field(default_factory=list)
    fingerprint: str = ""  # content hash for dedup

    @property
    def is_valid(self) -> bool:
        """True if code parsed without errors."""
        return len(self.parse_errors) == 0

    @property
    def exported_symbols(self) -> List[Symbol]:
        """Symbols that are part of the public API."""
        return [s for s in self.symbols if s.is_exported]

    @property
    def function_count(self) -> int:
        return sum(
            1
            for s in self.symbols
            if s.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)
        )

    @property
    def class_count(self) -> int:
        return sum(1 for s in self.symbols if s.kind == SymbolKind.CLASS)

    def to_dict(self) -> Dict:
        return {
            "language": self.language,
            "symbols": [s.to_dict() for s in self.symbols],
            "imports": [i.to_dict() for i in self.imports],
            "complexity": self.complexity.to_dict(),
            "parse_errors": self.parse_errors,
            "fingerprint": self.fingerprint,
            "is_valid": self.is_valid,
            "function_count": self.function_count,
            "class_count": self.class_count,
        }

    def to_repo_map(self) -> str:
        """Compact repo-map string (Aider-style) for context injection."""
        lines: List[str] = []
        for s in self.symbols:
            indent = "  " if s.parent else ""
            if s.kind == SymbolKind.CLASS:
                lines.append(f"{indent}class {s.name}:")
            elif s.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                sig = s.signature or f"def {s.name}(...)"
                lines.append(f"{indent}{sig}")
            elif s.kind == SymbolKind.INTERFACE:
                lines.append(f"{indent}interface {s.name}")
            elif s.kind == SymbolKind.TYPE_ALIAS:
                lines.append(f"{indent}type {s.name}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

# Regex patterns for language detection from code content
_LANG_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("python", re.compile(r"^\s*(def|class|import|from|if __name__)\b", re.M)),
    ("python", re.compile(r"^\s*@\w+\s*\n\s*def\s", re.M)),
    ("typescript", re.compile(r"\b(interface|type|enum)\s+\w+\s*[{=]", re.M)),
    ("typescript", re.compile(r":\s*(string|number|boolean|void)\b")),
    ("javascript", re.compile(r"\b(const|let|var)\s+\w+\s*=\s*(function|=>|\()", re.M)),
    ("javascript", re.compile(r"\bfunction\s+\w+\s*\(", re.M)),
    ("javascript", re.compile(r"\bexport\s+(default\s+)?(function|class|const)", re.M)),
    ("rust", re.compile(r"\b(fn|impl|struct|enum|trait|pub|mod)\s+\w+", re.M)),
    ("go", re.compile(r"\b(func|package|import)\s+", re.M)),
]

# Fenced code block language hints
_FENCE_LANG_MAP = {
    "python": "python",
    "py": "python",
    "python3": "python",
    "javascript": "javascript",
    "js": "javascript",
    "jsx": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "tsx": "tsx",
    "rust": "rust",
    "rs": "rust",
    "go": "go",
    "golang": "go",
}


def detect_language(code: str, hint: str = "") -> str:
    """
    Detect programming language from code content.

    Parameters
    ----------
    code : str
        The source code to analyze.
    hint : str
        Optional language hint (e.g., from fenced code block tag).

    Returns
    -------
    str
        Detected language name or "unknown".
    """
    # Use hint if provided
    if hint:
        normalized = hint.lower().strip()
        if normalized in _FENCE_LANG_MAP:
            return _FENCE_LANG_MAP[normalized]

    # Score each language by pattern matches
    scores: Dict[str, int] = {}
    for lang, pattern in _LANG_PATTERNS:
        if pattern.search(code):
            scores[lang] = scores.get(lang, 0) + 1

    if not scores:
        return "unknown"

    return max(scores, key=scores.get)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Code block extraction (enhanced from measure.py)
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(\w+)?\s*\n(.*?)```", re.S)


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Extract fenced code blocks from markdown text.

    Returns
    -------
    list of (language_hint, code) tuples.
    """
    blocks: List[Tuple[str, str]] = []
    for match in _CODE_BLOCK_RE.finditer(text):
        lang_hint = match.group(1) or ""
        code = match.group(2).strip()
        if code:
            blocks.append((lang_hint, code))
    return blocks


# ---------------------------------------------------------------------------
# Python AST extraction (stdlib ast — deepest analysis)
# ---------------------------------------------------------------------------


class _PythonVisitor(ast.NodeVisitor):
    """Walk a Python AST and extract symbols, complexity, and metrics."""

    def __init__(self) -> None:
        self.symbols: List[Symbol] = []
        self.imports: List[ImportInfo] = []
        self.complexity = 1  # base McCabe complexity
        self.max_depth = 0
        self._depth = 0
        self._parent_stack: List[str] = []
        # Counters for metrics
        self._func_count = 0
        self._func_with_annotations = 0
        self._func_with_docstrings = 0
        self._class_count = 0
        self._class_with_docstrings = 0
        self._assertion_count = 0
        self._test_func_count = 0
        # Anti-patterns
        self.bare_except_count = 0
        self.broad_except_count = 0
        self.eval_exec_count = 0
        self.wildcard_import_count = 0
        self.empty_block_count = 0

    @property
    def _parent(self) -> str:
        return self._parent_stack[-1] if self._parent_stack else ""

    def _push_depth(self) -> None:
        self._depth += 1
        self.max_depth = max(self.max_depth, self._depth)

    def _pop_depth(self) -> None:
        self._depth = max(0, self._depth - 1)

    def _get_docstring(self, node: ast.AST) -> str:
        """Extract first line of docstring from a function or class."""
        try:
            ds = ast.get_docstring(node)
            if ds:
                first_line = ds.split("\n")[0].strip()
                return first_line[:200]  # cap at 200 chars
        except Exception:
            pass
        return ""

    def _format_annotation(self, node: Optional[ast.AST]) -> str:
        """Convert an annotation AST node to a readable string."""
        if node is None:
            return ""
        try:
            return ast.unparse(node)
        except Exception:
            return ""

    def _build_signature(self, node: ast.FunctionDef) -> str:
        """Build a function signature string."""
        try:
            # Build args
            args_parts: List[str] = []
            all_args = node.args

            # Positional args
            defaults_offset = len(all_args.args) - len(all_args.defaults)
            for i, arg in enumerate(all_args.args):
                part = arg.arg
                if arg.annotation:
                    part += f": {self._format_annotation(arg.annotation)}"
                if i >= defaults_offset:
                    default = all_args.defaults[i - defaults_offset]
                    try:
                        part += f" = {ast.unparse(default)}"
                    except Exception:
                        part += " = ..."
                args_parts.append(part)

            # *args
            if all_args.vararg:
                va = f"*{all_args.vararg.arg}"
                if all_args.vararg.annotation:
                    va += f": {self._format_annotation(all_args.vararg.annotation)}"
                args_parts.append(va)

            # **kwargs
            if all_args.kwarg:
                kw = f"**{all_args.kwarg.arg}"
                if all_args.kwarg.annotation:
                    kw += f": {self._format_annotation(all_args.kwarg.annotation)}"
                args_parts.append(kw)

            args_str = ", ".join(args_parts)

            # Filter 'self' and 'cls' from display
            if args_str.startswith("self, "):
                args_str = args_str[6:]
            elif args_str.startswith("cls, "):
                args_str = args_str[5:]
            elif args_str == "self" or args_str == "cls":
                args_str = ""

            ret = ""
            if node.returns:
                ret = f" -> {self._format_annotation(node.returns)}"

            return f"def {node.name}({args_str}){ret}"
        except Exception:
            return f"def {node.name}(...)"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._func_count += 1
        is_method = bool(self._parent_stack)
        kind = SymbolKind.METHOD if is_method else SymbolKind.FUNCTION

        # Check for property decorator
        decorators = []
        for dec in node.decorator_list:
            try:
                dec_name = ast.unparse(dec)
                decorators.append(dec_name)
                if dec_name == "property":
                    kind = SymbolKind.PROPERTY
            except Exception:
                pass

        docstring = self._get_docstring(node)
        if docstring:
            self._func_with_docstrings += 1

        # Type annotation coverage
        has_return_ann = node.returns is not None
        has_any_ann = has_return_ann or any(
            a.annotation is not None
            for a in node.args.args
            if a.arg not in ("self", "cls")
        )
        if has_any_ann:
            self._func_with_annotations += 1

        # Test detection
        if node.name.startswith("test_"):
            self._test_func_count += 1

        signature = self._build_signature(node)
        return_type = self._format_annotation(node.returns)

        self.symbols.append(
            Symbol(
                name=node.name,
                kind=kind,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature=signature,
                docstring=docstring,
                parent=self._parent,
                decorators=decorators,
                is_exported=not node.name.startswith("_"),
                type_annotation=return_type,
            )
        )

        # Recurse into function body
        self._parent_stack.append(node.name)
        self._push_depth()
        self.generic_visit(node)
        self._pop_depth()
        self._parent_stack.pop()

    # Alias for async functions
    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_count += 1

        decorators = []
        for dec in node.decorator_list:
            try:
                decorators.append(ast.unparse(dec))
            except Exception:
                pass

        docstring = self._get_docstring(node)
        if docstring:
            self._class_with_docstrings += 1

        # Build a signature with base classes
        bases: List[str] = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                bases.append("?")
        sig = f"class {node.name}"
        if bases:
            sig += f"({', '.join(bases)})"

        self.symbols.append(
            Symbol(
                name=node.name,
                kind=SymbolKind.CLASS,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature=sig,
                docstring=docstring,
                parent=self._parent,
                decorators=decorators,
                is_exported=not node.name.startswith("_"),
            )
        )

        # Recurse into class body
        self._parent_stack.append(node.name)
        self._push_depth()
        self.generic_visit(node)
        self._pop_depth()
        self._parent_stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    module=alias.name,
                    alias=alias.asname or "",
                    line=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        names = []
        is_wildcard = False
        for alias in node.names:
            if alias.name == "*":
                is_wildcard = True
                self.wildcard_import_count += 1
            else:
                names.append(alias.name)

        self.imports.append(
            ImportInfo(
                module=module,
                names=names,
                is_relative=bool(node.level and node.level > 0),
                line=node.lineno,
                is_wildcard=is_wildcard,
            )
        )
        self.generic_visit(node)

    # --- Complexity tracking ---

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self._push_depth()
        self.generic_visit(node)
        self._pop_depth()

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self._push_depth()
        self.generic_visit(node)
        self._pop_depth()

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self._push_depth()
        self.generic_visit(node)
        self._pop_depth()

    def visit_With(self, node: ast.With) -> None:
        self._push_depth()
        self.generic_visit(node)
        self._pop_depth()

    visit_AsyncWith = visit_With

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        # Detect bare except (no exception type)
        if node.type is None:
            self.bare_except_count += 1
        else:
            # Detect broad except (except Exception)
            try:
                name = ast.unparse(node.type)
                if name in ("Exception", "BaseException"):
                    self.broad_except_count += 1
            except Exception:
                pass

        # Detect empty except blocks (just 'pass')
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self.empty_block_count += 1

        self._push_depth()
        self.generic_visit(node)
        self._pop_depth()

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # Each 'and' / 'or' adds a path
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        # Ternary expression
        self.complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self._assertion_count += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Detect eval() and exec()
        try:
            if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                self.eval_exec_count += 1
        except Exception:
            pass
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Track module-level variable assignments
        if not self._parent_stack:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Detect constants (ALL_CAPS)
                    is_const = target.id.isupper() and "_" in target.id
                    kind = SymbolKind.CONSTANT if is_const else SymbolKind.VARIABLE
                    self.symbols.append(
                        Symbol(
                            name=target.id,
                            kind=kind,
                            line=node.lineno,
                            is_exported=not target.id.startswith("_"),
                        )
                    )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Type-annotated assignments at module level
        if not self._parent_stack and isinstance(node.target, ast.Name):
            ann = self._format_annotation(node.annotation)
            is_const = node.target.id.isupper() and "_" in node.target.id
            kind = SymbolKind.CONSTANT if is_const else SymbolKind.VARIABLE
            self.symbols.append(
                Symbol(
                    name=node.target.id,
                    kind=kind,
                    line=node.lineno,
                    is_exported=not node.target.id.startswith("_"),
                    type_annotation=ann,
                )
            )
        self.generic_visit(node)

    def build_metrics(self, code: str) -> ComplexityMetrics:
        """Build complexity metrics from accumulated visitor state."""
        lines = code.split("\n")
        num_lines = len(lines)
        num_blank = sum(1 for l in lines if not l.strip())
        num_comment = sum(1 for l in lines if l.strip().startswith("#"))

        total_documentable = self._func_count + self._class_count
        docstring_coverage = (
            (self._func_with_docstrings + self._class_with_docstrings)
            / total_documentable
            if total_documentable > 0
            else 0.0
        )

        annotation_coverage = (
            self._func_with_annotations / self._func_count
            if self._func_count > 0
            else 0.0
        )

        assertion_density = (
            self._assertion_count / self._func_count if self._func_count > 0 else 0.0
        )

        return ComplexityMetrics(
            cyclomatic_complexity=self.complexity,
            max_nesting_depth=self.max_depth,
            num_functions=self._func_count,
            num_classes=self._class_count,
            num_imports=len(self.imports),
            num_lines=num_lines,
            num_blank_lines=num_blank,
            num_comment_lines=num_comment,
            has_type_annotations=self._func_with_annotations > 0,
            type_annotation_coverage=annotation_coverage,
            has_docstrings=self._func_with_docstrings > 0
            or self._class_with_docstrings > 0,
            docstring_coverage=docstring_coverage,
            has_tests=self._test_func_count > 0,
            assertion_density=assertion_density,
            bare_except_count=self.bare_except_count,
            broad_except_count=self.broad_except_count,
            eval_exec_count=self.eval_exec_count,
            wildcard_import_count=self.wildcard_import_count,
            empty_block_count=self.empty_block_count,
        )


def _analyze_python_stdlib(code: str) -> CodeAnalysis:
    """Analyze Python code using stdlib ast module (deepest analysis)."""
    errors: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        errors.append(f"SyntaxError: {exc.msg} (line {exc.lineno})")
        return CodeAnalysis(
            language="python",
            parse_errors=errors,
            fingerprint=_fingerprint(code),
        )

    visitor = _PythonVisitor()
    visitor.visit(tree)

    return CodeAnalysis(
        language="python",
        symbols=visitor.symbols,
        imports=visitor.imports,
        complexity=visitor.build_metrics(code),
        parse_errors=errors,
        fingerprint=_fingerprint(code),
    )


# ---------------------------------------------------------------------------
# Tree-sitter extraction (JavaScript / TypeScript)
# ---------------------------------------------------------------------------


def _ts_extract_node_text(node, source_bytes: bytes) -> str:
    """Get the text of a tree-sitter node."""
    return source_bytes[node.start_byte : node.end_byte].decode(
        "utf-8", errors="replace"
    )


def _analyze_js_ts(code: str, language: str) -> CodeAnalysis:
    """Analyze JavaScript/TypeScript code using tree-sitter."""
    ts_lang = language
    if language == "javascript":
        ts_lang = "javascript"
    elif language in ("typescript", "tsx"):
        ts_lang = language

    tree = _ts_parse(code, ts_lang)
    if tree is None:
        # Fallback: basic regex extraction
        return _analyze_regex_fallback(code, language)

    source_bytes = code.encode("utf-8")
    root = tree.root_node

    symbols: List[Symbol] = []
    imports: List[ImportInfo] = []
    errors: List[str] = []
    complexity = 1
    max_depth = 0

    # Check for parse errors
    if root.has_error:
        errors.append("Parse error detected in source")

    def _walk(node, depth: int = 0, parent_name: str = "") -> None:
        nonlocal complexity, max_depth
        max_depth = max(max_depth, depth)

        node_type = node.type

        # --- Function declarations ---
        if node_type in (
            "function_declaration",
            "method_definition",
            "arrow_function",
            "function",
        ):
            name = ""
            for child in node.children:
                if child.type == "identifier":
                    name = _ts_extract_node_text(child, source_bytes)
                    break
                elif child.type == "property_identifier":
                    name = _ts_extract_node_text(child, source_bytes)
                    break

            # For arrow functions assigned to variables, look at parent
            if not name and node.parent:
                p = node.parent
                if p.type == "variable_declarator":
                    for child in p.children:
                        if child.type == "identifier":
                            name = _ts_extract_node_text(child, source_bytes)
                            break

            if name:
                kind = SymbolKind.METHOD if parent_name else SymbolKind.FUNCTION
                sig = _ts_extract_node_text(node, source_bytes).split("{")[0].strip()
                if len(sig) > 200:
                    sig = sig[:200] + "..."
                symbols.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        signature=sig,
                        parent=parent_name,
                        is_exported=not name.startswith("_"),
                    )
                )

        # --- Class declarations ---
        elif node_type == "class_declaration":
            name = ""
            for child in node.children:
                if child.type == "identifier" or child.type == "type_identifier":
                    name = _ts_extract_node_text(child, source_bytes)
                    break
            if name:
                symbols.append(
                    Symbol(
                        name=name,
                        kind=SymbolKind.CLASS,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parent=parent_name,
                        is_exported=not name.startswith("_"),
                    )
                )
                # Recurse into class body with class as parent
                for child in node.children:
                    _walk(child, depth + 1, parent_name=name)
                return  # Don't double-visit children

        # --- Interface/Type declarations (TypeScript) ---
        elif node_type in ("interface_declaration", "type_alias_declaration"):
            name = ""
            for child in node.children:
                if child.type == "type_identifier" or child.type == "identifier":
                    name = _ts_extract_node_text(child, source_bytes)
                    break
            if name:
                kind = (
                    SymbolKind.INTERFACE
                    if node_type == "interface_declaration"
                    else SymbolKind.TYPE_ALIAS
                )
                symbols.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parent=parent_name,
                        is_exported=not name.startswith("_"),
                    )
                )

        # --- Import statements ---
        elif node_type == "import_statement":
            text = _ts_extract_node_text(node, source_bytes)
            # Parse import source
            source_node = None
            for child in node.children:
                if child.type == "string":
                    source_node = child
                    break
            if source_node:
                module = _ts_extract_node_text(source_node, source_bytes).strip("'\"")
                # Extract imported names
                names: List[str] = []
                for child in node.children:
                    if child.type == "import_clause":
                        for sub in child.children:
                            if sub.type == "identifier":
                                names.append(_ts_extract_node_text(sub, source_bytes))
                            elif sub.type == "named_imports":
                                for spec in sub.children:
                                    if spec.type == "import_specifier":
                                        for n in spec.children:
                                            if n.type == "identifier":
                                                names.append(
                                                    _ts_extract_node_text(
                                                        n, source_bytes
                                                    )
                                                )
                                                break
                imports.append(
                    ImportInfo(
                        module=module,
                        names=names,
                        line=node.start_point[0] + 1,
                        is_relative=module.startswith("."),
                    )
                )

        # --- Complexity tracking ---
        elif node_type in (
            "if_statement",
            "for_statement",
            "for_in_statement",
            "while_statement",
            "catch_clause",
            "ternary_expression",
        ):
            complexity += 1

        elif node_type in ("binary_expression",):
            # Check for && and ||
            for child in node.children:
                if child.type in ("&&", "||"):
                    complexity += 1

        # Recurse into children
        for child in node.children:
            _walk(child, depth + 1, parent_name)

    _walk(root)

    # Build metrics
    lines = code.split("\n")
    metrics = ComplexityMetrics(
        cyclomatic_complexity=complexity,
        max_nesting_depth=max_depth,
        num_functions=sum(
            1 for s in symbols if s.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)
        ),
        num_classes=sum(1 for s in symbols if s.kind == SymbolKind.CLASS),
        num_imports=len(imports),
        num_lines=len(lines),
        num_blank_lines=sum(1 for l in lines if not l.strip()),
        num_comment_lines=sum(
            1 for l in lines if l.strip().startswith("//") or l.strip().startswith("/*")
        ),
    )

    return CodeAnalysis(
        language=language,
        symbols=symbols,
        imports=imports,
        complexity=metrics,
        parse_errors=errors,
        fingerprint=_fingerprint(code),
    )


# ---------------------------------------------------------------------------
# Regex fallback (for unsupported languages or missing tree-sitter)
# ---------------------------------------------------------------------------

_FUNC_RE = re.compile(
    r"^\s*(?:(?:pub|pub\s*\(crate\)|async|export|default)\s+)*"
    r"(?:def|fn|func|function)\s+(\w+)\s*\(",
    re.M,
)
_CLASS_RE = re.compile(
    r"^\s*(?:(?:pub|export|abstract)\s+)*(?:class|struct|interface|trait|enum)\s+(\w+)",
    re.M,
)
_IMPORT_RE = re.compile(r"^\s*(?:import|from|use|require)\s+(.+?)(?:;|\s*$)", re.M)


def _analyze_regex_fallback(code: str, language: str) -> CodeAnalysis:
    """Basic regex extraction when AST parsing is unavailable."""
    symbols: List[Symbol] = []
    imports: List[ImportInfo] = []

    for match in _FUNC_RE.finditer(code):
        name = match.group(1)
        line = code[: match.start()].count("\n") + 1
        symbols.append(
            Symbol(
                name=name,
                kind=SymbolKind.FUNCTION,
                line=line,
                is_exported=not name.startswith("_"),
            )
        )

    for match in _CLASS_RE.finditer(code):
        name = match.group(1)
        line = code[: match.start()].count("\n") + 1
        symbols.append(
            Symbol(
                name=name,
                kind=SymbolKind.CLASS,
                line=line,
                is_exported=not name.startswith("_"),
            )
        )

    for match in _IMPORT_RE.finditer(code):
        module = match.group(1).strip()
        line = code[: match.start()].count("\n") + 1
        imports.append(ImportInfo(module=module, line=line))

    lines = code.split("\n")
    metrics = ComplexityMetrics(
        num_functions=len([s for s in symbols if s.kind == SymbolKind.FUNCTION]),
        num_classes=len([s for s in symbols if s.kind == SymbolKind.CLASS]),
        num_imports=len(imports),
        num_lines=len(lines),
        num_blank_lines=sum(1 for l in lines if not l.strip()),
    )

    return CodeAnalysis(
        language=language,
        symbols=symbols,
        imports=imports,
        complexity=metrics,
        fingerprint=_fingerprint(code),
    )


# ---------------------------------------------------------------------------
# Content fingerprinting
# ---------------------------------------------------------------------------


def _fingerprint(code: str) -> str:
    """SHA-256 fingerprint for content deduplication."""
    # Normalize whitespace for comparison
    normalized = re.sub(r"\s+", " ", code.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_code(code: str, language: str = "", hint: str = "") -> CodeAnalysis:
    """
    Full AST analysis of a code block.

    Parameters
    ----------
    code : str
        Source code to analyze.
    language : str
        Language name. If empty, auto-detected.
    hint : str
        Optional language hint (from fenced code block).

    Returns
    -------
    CodeAnalysis
        Complete analysis with symbols, imports, complexity, and metrics.
    """
    if not code or not code.strip():
        return CodeAnalysis(language=language or "unknown")

    if not language:
        language = detect_language(code, hint=hint)

    if language == "python":
        return _analyze_python_stdlib(code)
    elif language in ("javascript", "typescript", "tsx"):
        return _analyze_js_ts(code, language)
    else:
        return _analyze_regex_fallback(code, language)


def extract_symbols(code: str, language: str = "") -> List[Symbol]:
    """Extract symbols (functions, classes, etc.) from code."""
    analysis = analyze_code(code, language)
    return analysis.symbols


def extract_complexity(code: str, language: str = "") -> ComplexityMetrics:
    """Extract complexity metrics from code."""
    analysis = analyze_code(code, language)
    return analysis.complexity


def extract_dependencies(code: str, language: str = "") -> List[ImportInfo]:
    """Extract import/dependency information from code."""
    analysis = analyze_code(code, language)
    return analysis.imports


def analyze_response(text: str) -> List[CodeAnalysis]:
    """
    Extract and analyze all code blocks from an LLM response.

    This is the main integration point for the after-pipeline.
    Extracts fenced code blocks, detects language, and runs full
    AST analysis on each block.

    Parameters
    ----------
    text : str
        Full LLM response text (may contain markdown code blocks).

    Returns
    -------
    list[CodeAnalysis]
        Analysis results for each code block found.
    """
    blocks = extract_code_blocks(text)
    if not blocks:
        # Try analyzing the full text as code
        lang = detect_language(text)
        if lang != "unknown":
            analysis = analyze_code(text, lang)
            if analysis.symbols or analysis.imports:
                return [analysis]
        return []

    results: List[CodeAnalysis] = []
    for hint, code in blocks:
        analysis = analyze_code(code, hint=hint)
        results.append(analysis)

    return results
