"""Tests for engram.extraction.ast_engine — AST-based code analysis."""

import pytest

from engram.extraction.ast_engine import (
    CodeAnalysis,
    ComplexityMetrics,
    ImportInfo,
    Symbol,
    SymbolKind,
    analyze_code,
    analyze_response,
    detect_language,
    extract_code_blocks,
    extract_complexity,
    extract_dependencies,
    extract_symbols,
)


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    def test_python_def(self):
        assert detect_language("def hello(): pass") == "python"

    def test_python_class(self):
        assert detect_language("class Foo:\n    pass") == "python"

    def test_python_import(self):
        assert detect_language("import os\nfrom typing import List") == "python"

    def test_python_decorator(self):
        assert detect_language("@dataclass\ndef foo(): pass") == "python"

    def test_javascript_function(self):
        assert detect_language("function hello() { return 1; }") == "javascript"

    def test_javascript_const_arrow(self):
        assert detect_language("const foo = () => {};") == "javascript"

    def test_javascript_export(self):
        assert detect_language("export default function App() {}") == "javascript"

    def test_typescript_interface(self):
        assert detect_language("interface Props { name: string; }") == "typescript"

    def test_typescript_type(self):
        assert detect_language("type Result = { ok: boolean }") == "typescript"

    def test_hint_overrides(self):
        assert detect_language("any code", hint="py") == "python"
        assert detect_language("any code", hint="javascript") == "javascript"
        assert detect_language("any code", hint="ts") == "typescript"
        assert detect_language("any code", hint="tsx") == "tsx"

    def test_unknown_code(self):
        assert detect_language("hello world") == "unknown"

    def test_empty_code(self):
        assert detect_language("") == "unknown"


# ---------------------------------------------------------------------------
# Code block extraction
# ---------------------------------------------------------------------------


class TestExtractCodeBlocks:
    def test_single_block(self):
        text = "Here's code:\n```python\ndef foo(): pass\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == "python"
        assert "def foo" in blocks[0][1]

    def test_multiple_blocks(self):
        text = (
            "```python\ndef a(): pass\n```\nText\n```javascript\nfunction b() {}\n```"
        )
        blocks = extract_code_blocks(text)
        assert len(blocks) == 2
        assert blocks[0][0] == "python"
        assert blocks[1][0] == "javascript"

    def test_no_language_hint(self):
        text = "```\ndef foo(): pass\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == ""

    def test_no_code_blocks(self):
        blocks = extract_code_blocks("Just regular text.")
        assert len(blocks) == 0


# ---------------------------------------------------------------------------
# Python AST analysis
# ---------------------------------------------------------------------------


class TestPythonAnalysis:
    SAMPLE_CODE = """
import os
from typing import List, Optional

class UserService:
    \"\"\"Manages user operations.\"\"\"
    
    def __init__(self, db: object) -> None:
        self.db = db
    
    def get_user(self, user_id: int) -> Optional[dict]:
        \"\"\"Fetch a user by ID.\"\"\"
        return self.db.get(user_id)
    
    def list_users(self) -> List[dict]:
        return self.db.all()

def helper(x: int) -> int:
    if x > 0:
        return x * 2
    return 0

MAX_RETRIES = 3
"""

    def test_basic_parse(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        assert analysis.language == "python"
        assert analysis.is_valid
        assert len(analysis.parse_errors) == 0

    def test_symbol_count(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        # UserService, __init__, get_user, list_users, helper, MAX_RETRIES
        assert len(analysis.symbols) >= 5

    def test_class_extraction(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        classes = [s for s in analysis.symbols if s.kind == SymbolKind.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "UserService"
        assert classes[0].docstring == "Manages user operations."

    def test_method_extraction(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        methods = [s for s in analysis.symbols if s.kind == SymbolKind.METHOD]
        names = {m.name for m in methods}
        assert "__init__" in names
        assert "get_user" in names
        assert "list_users" in names

    def test_method_parent(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        get_user = next(s for s in analysis.symbols if s.name == "get_user")
        assert get_user.parent == "UserService"
        assert get_user.qualified_name == "UserService.get_user"

    def test_function_extraction(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        funcs = [s for s in analysis.symbols if s.kind == SymbolKind.FUNCTION]
        assert any(f.name == "helper" for f in funcs)

    def test_constant_extraction(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        consts = [s for s in analysis.symbols if s.kind == SymbolKind.CONSTANT]
        assert any(c.name == "MAX_RETRIES" for c in consts)

    def test_signature_generation(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        helper = next(s for s in analysis.symbols if s.name == "helper")
        assert "def helper(x: int) -> int" == helper.signature

    def test_self_filtered_from_signature(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        get_user = next(s for s in analysis.symbols if s.name == "get_user")
        assert "self" not in get_user.signature
        assert "user_id: int" in get_user.signature

    def test_imports(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        assert len(analysis.imports) == 2
        modules = {i.module for i in analysis.imports}
        assert "os" in modules
        assert "typing" in modules

    def test_import_names(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        typing_imp = next(i for i in analysis.imports if i.module == "typing")
        assert "List" in typing_imp.names
        assert "Optional" in typing_imp.names

    def test_complexity(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        # At least 1 (base) + 1 (if in helper)
        assert analysis.complexity.cyclomatic_complexity >= 2

    def test_type_annotation_coverage(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        # All functions have annotations
        assert analysis.complexity.type_annotation_coverage > 0.5

    def test_docstring_coverage(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        assert analysis.complexity.has_docstrings

    def test_exported_symbols(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        exported = analysis.exported_symbols
        # __init__ starts with _ so not exported
        names = {s.name for s in exported}
        assert "UserService" in names
        assert "helper" in names
        assert "__init__" not in names

    def test_fingerprint(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        assert len(analysis.fingerprint) == 16  # 16 hex chars

    def test_fingerprint_stability(self):
        a1 = analyze_code(self.SAMPLE_CODE, "python")
        a2 = analyze_code(self.SAMPLE_CODE, "python")
        assert a1.fingerprint == a2.fingerprint

    def test_fingerprint_changes_with_content(self):
        a1 = analyze_code("def foo(): pass", "python")
        a2 = analyze_code("def bar(): pass", "python")
        assert a1.fingerprint != a2.fingerprint

    def test_repo_map_generation(self):
        analysis = analyze_code(self.SAMPLE_CODE, "python")
        repo_map = analysis.to_repo_map()
        assert "class UserService" in repo_map
        assert "def helper" in repo_map

    def test_syntax_error(self):
        analysis = analyze_code("def foo(\n  broken", "python")
        assert not analysis.is_valid
        assert len(analysis.parse_errors) > 0
        assert "SyntaxError" in analysis.parse_errors[0]


class TestPythonComplexity:
    def test_bare_except_detection(self):
        code = """
try:
    risky()
except:
    pass
"""
        analysis = analyze_code(code, "python")
        assert analysis.complexity.bare_except_count == 1
        assert analysis.complexity.empty_block_count == 1

    def test_broad_except_detection(self):
        code = """
try:
    risky()
except Exception:
    handle()
"""
        analysis = analyze_code(code, "python")
        assert analysis.complexity.broad_except_count == 1

    def test_eval_exec_detection(self):
        code = """
result = eval(user_input)
exec(code_string)
"""
        analysis = analyze_code(code, "python")
        assert analysis.complexity.eval_exec_count == 2

    def test_wildcard_import_detection(self):
        code = "from os import *\nfrom sys import *"
        analysis = analyze_code(code, "python")
        assert analysis.complexity.wildcard_import_count == 2

    def test_nesting_depth(self):
        code = """
def deep():
    if True:
        for x in range(10):
            while True:
                if x > 5:
                    pass
"""
        analysis = analyze_code(code, "python")
        assert analysis.complexity.max_nesting_depth >= 4

    def test_boolean_operator_complexity(self):
        code = """
def check(a, b, c):
    if a and b or c:
        return True
"""
        analysis = analyze_code(code, "python")
        # base(1) + if(1) + and(1) + or — exact count depends on AST structure
        assert analysis.complexity.cyclomatic_complexity >= 3

    def test_test_detection(self):
        code = """
def test_something():
    assert True

def test_other():
    assert 1 == 1
"""
        analysis = analyze_code(code, "python")
        assert analysis.complexity.has_tests
        assert analysis.complexity.assertion_density > 0

    def test_line_counts(self):
        code = "# comment\ndef foo():\n    pass\n\n"
        analysis = analyze_code(code, "python")
        assert analysis.complexity.num_lines >= 3
        assert analysis.complexity.num_comment_lines >= 1
        assert analysis.complexity.num_blank_lines >= 1

    def test_async_function(self):
        code = """
async def fetch_data(url: str) -> dict:
    \"\"\"Fetch data from URL.\"\"\"
    pass
"""
        analysis = analyze_code(code, "python")
        funcs = [s for s in analysis.symbols if s.kind == SymbolKind.FUNCTION]
        assert any(f.name == "fetch_data" for f in funcs)

    def test_decorator_extraction(self):
        code = """
@staticmethod
def my_func():
    pass

@property
def my_prop(self):
    return self._val
"""
        analysis = analyze_code(code, "python")
        my_func = next(s for s in analysis.symbols if s.name == "my_func")
        assert "staticmethod" in my_func.decorators

    def test_class_with_bases(self):
        code = """
class MyError(ValueError, RuntimeError):
    pass
"""
        analysis = analyze_code(code, "python")
        cls = next(s for s in analysis.symbols if s.name == "MyError")
        assert "ValueError" in cls.signature
        assert "RuntimeError" in cls.signature


# ---------------------------------------------------------------------------
# JavaScript/TypeScript analysis
# ---------------------------------------------------------------------------


class TestJavaScriptAnalysis:
    def test_function_extraction(self):
        code = "function hello() { return 'hi'; }"
        analysis = analyze_code(code, "javascript")
        funcs = [s for s in analysis.symbols if s.kind == SymbolKind.FUNCTION]
        assert any(f.name == "hello" for f in funcs)

    def test_class_extraction(self):
        code = """
class UserService {
    constructor(db) {
        this.db = db;
    }
    getUser(id) {
        return this.db.get(id);
    }
}
"""
        analysis = analyze_code(code, "javascript")
        classes = [s for s in analysis.symbols if s.kind == SymbolKind.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "UserService"

    def test_import_extraction(self):
        code = "import React from 'react';\nimport { useState } from 'react';"
        analysis = analyze_code(code, "javascript")
        assert len(analysis.imports) >= 1
        modules = {i.module for i in analysis.imports}
        assert "react" in modules

    def test_complexity_tracking(self):
        code = """
function complex(x) {
    if (x > 0) {
        for (let i = 0; i < x; i++) {
            if (i % 2 === 0) {
                return i;
            }
        }
    }
    return 0;
}
"""
        analysis = analyze_code(code, "javascript")
        assert analysis.complexity.cyclomatic_complexity >= 3


class TestTypeScriptAnalysis:
    def test_interface_extraction(self):
        code = "interface Props { name: string; age: number; }"
        analysis = analyze_code(code, "typescript")
        interfaces = [s for s in analysis.symbols if s.kind == SymbolKind.INTERFACE]
        assert len(interfaces) == 1
        assert interfaces[0].name == "Props"

    def test_type_alias_extraction(self):
        code = "type Result = { ok: boolean; data: string; }"
        analysis = analyze_code(code, "typescript")
        types = [s for s in analysis.symbols if s.kind == SymbolKind.TYPE_ALIAS]
        assert len(types) == 1
        assert types[0].name == "Result"


# ---------------------------------------------------------------------------
# Regex fallback
# ---------------------------------------------------------------------------


class TestRegexFallback:
    def test_unknown_language(self):
        code = """
fn main() {
    println!("hello");
}

struct Config {
    name: String,
}
"""
        analysis = analyze_code(code, "rust")
        # Should use regex fallback
        assert analysis.language == "rust"
        funcs = [s for s in analysis.symbols if s.kind == SymbolKind.FUNCTION]
        assert any(f.name == "main" for f in funcs)
        classes = [s for s in analysis.symbols if s.kind == SymbolKind.CLASS]
        assert any(c.name == "Config" for c in classes)


# ---------------------------------------------------------------------------
# analyze_response (markdown extraction)
# ---------------------------------------------------------------------------


class TestAnalyzeResponse:
    def test_single_block(self):
        text = "```python\ndef foo(): pass\n```"
        results = analyze_response(text)
        assert len(results) == 1
        assert results[0].language == "python"

    def test_multiple_blocks(self):
        text = "```python\ndef a(): pass\n```\n```javascript\nfunction b() {}\n```"
        results = analyze_response(text)
        assert len(results) == 2

    def test_no_code_blocks(self):
        results = analyze_response("Just regular text with no code.")
        assert len(results) == 0

    def test_plain_code_detection(self):
        results = analyze_response("def foo(x: int) -> int:\n    return x * 2")
        assert len(results) == 1
        assert results[0].language == "python"

    def test_code_with_surrounding_text(self):
        text = """Here's the implementation:

```python
class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
```

This handles basic addition.
"""
        results = analyze_response(text)
        assert len(results) == 1
        assert any(s.name == "Calculator" for s in results[0].symbols)
        assert any(s.name == "add" for s in results[0].symbols)


# ---------------------------------------------------------------------------
# Public API convenience functions
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_extract_symbols(self):
        symbols = extract_symbols("def foo(): pass\nclass Bar: pass")
        names = {s.name for s in symbols}
        assert "foo" in names
        assert "Bar" in names

    def test_extract_complexity(self):
        metrics = extract_complexity("def foo():\n    if True:\n        pass")
        assert isinstance(metrics, ComplexityMetrics)
        assert metrics.cyclomatic_complexity >= 2

    def test_extract_dependencies(self):
        deps = extract_dependencies("import os\nfrom sys import argv")
        modules = {d.module for d in deps}
        assert "os" in modules
        assert "sys" in modules


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_code(self):
        analysis = analyze_code("", "python")
        assert analysis.language == "python"
        assert len(analysis.symbols) == 0

    def test_whitespace_only(self):
        analysis = analyze_code("   \n\n  ", "python")
        assert len(analysis.symbols) == 0

    def test_comment_only(self):
        analysis = analyze_code("# Just a comment\n# Another", "python")
        assert analysis.is_valid
        assert len(analysis.symbols) == 0

    def test_very_long_code(self):
        # Generate a large code block
        lines = [
            f"def func_{i}(x: int) -> int:\n    return x + {i}\n" for i in range(100)
        ]
        code = "\n".join(lines)
        analysis = analyze_code(code, "python")
        assert analysis.complexity.num_functions == 100

    def test_unicode_in_code(self):
        code = 'def greet(name: str) -> str:\n    return f"Hello, {name}!"'
        analysis = analyze_code(code, "python")
        assert analysis.is_valid

    def test_symbol_to_dict(self):
        sym = Symbol(
            name="foo",
            kind=SymbolKind.FUNCTION,
            line=1,
            signature="def foo(x: int) -> int",
        )
        d = sym.to_dict()
        assert d["name"] == "foo"
        assert d["kind"] == "function"
        assert d["line"] == 1

    def test_import_info_to_dict(self):
        imp = ImportInfo(module="os.path", names=["join", "dirname"], line=1)
        d = imp.to_dict()
        assert d["module"] == "os.path"
        assert "join" in d["names"]

    def test_code_analysis_to_dict(self):
        analysis = analyze_code("def foo(): pass", "python")
        d = analysis.to_dict()
        assert "language" in d
        assert "symbols" in d
        assert "complexity" in d
        assert "fingerprint" in d
