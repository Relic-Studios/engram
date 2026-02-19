"""
Tests for engram.extraction.fingerprint — Sentry-style error fingerprinting.

Covers:
  - Exception type extraction (Python, JS, generic, unknown)
  - Message normalization (paths, line numbers, variables, IPs, UUIDs, URLs, hex)
  - Stack frame extraction (Python frames, JS frames, stdlib filtering)
  - Fingerprint stability (same logical error → same hash)
  - Fingerprint differentiation (different errors → different hashes)
  - Full analyze_error() output structure
"""

from __future__ import annotations

import pytest

from engram.extraction.fingerprint import (
    analyze_error,
    compute_fingerprint,
    extract_exception_type,
    extract_frames,
    normalize_message,
)


# =========================================================================
# extract_exception_type
# =========================================================================


class TestExtractExceptionType:
    """Test exception type extraction from error messages."""

    def test_python_type_error(self):
        msg = "TypeError: unsupported operand type(s) for +: 'int' and 'str'"
        assert extract_exception_type(msg) == "TypeError"

    def test_python_import_error(self):
        msg = "ImportError: cannot import name 'foo' from 'bar.baz'"
        assert extract_exception_type(msg) == "ImportError"

    def test_python_value_error(self):
        msg = "ValueError: invalid literal for int() with base 10: 'abc'"
        assert extract_exception_type(msg) == "ValueError"

    def test_python_attribute_error(self):
        msg = "AttributeError: 'NoneType' object has no attribute 'split'"
        assert extract_exception_type(msg) == "AttributeError"

    def test_python_key_error(self):
        msg = "KeyError: 'missing_key'"
        assert extract_exception_type(msg) == "KeyError"

    def test_python_runtime_error(self):
        msg = "RuntimeError: maximum recursion depth exceeded"
        assert extract_exception_type(msg) == "RuntimeError"

    def test_js_type_error(self):
        msg = "TypeError: Cannot read properties of undefined (reading 'map')"
        assert extract_exception_type(msg) == "TypeError"

    def test_js_reference_error(self):
        msg = "ReferenceError: x is not defined"
        assert extract_exception_type(msg) == "ReferenceError"

    def test_js_syntax_error(self):
        msg = "SyntaxError: Unexpected token }"
        assert extract_exception_type(msg) == "SyntaxError"

    def test_js_range_error(self):
        msg = "RangeError: Maximum call stack size exceeded"
        assert extract_exception_type(msg) == "RangeError"

    def test_generic_traceback_extraction(self):
        msg = (
            "Traceback (most recent call last):\n"
            "  File 'app.py', line 10, in main\n"
            "    foo()\n"
            "  File 'app.py', line 5, in foo\n"
            "    raise ConnectionError('timeout')\n"
            "ConnectionError: timeout"
        )
        assert extract_exception_type(msg) == "ConnectionError"

    def test_unknown_error(self):
        msg = "Something went terribly wrong"
        assert extract_exception_type(msg) == "UnknownError"

    def test_empty_string(self):
        assert extract_exception_type("") == "UnknownError"

    def test_warning_type(self):
        msg = "DeprecationWarning: This function is deprecated"
        assert extract_exception_type(msg) == "DeprecationWarning"

    def test_custom_exception(self):
        msg = "DatabaseException: connection pool exhausted"
        assert extract_exception_type(msg) == "DatabaseException"


# =========================================================================
# normalize_message
# =========================================================================


class TestNormalizeMessage:
    """Test message template normalization."""

    def test_unix_path_replacement(self):
        msg = "File '/home/user/project/app.py', line 42"
        result = normalize_message(msg)
        assert "<PATH>" in result
        assert "/home/user" not in result

    def test_windows_path_replacement(self):
        msg = 'File "C:\\Users\\dev\\project\\main.py", line 10'
        result = normalize_message(msg)
        assert "<PATH>" in result
        assert "C:\\Users" not in result

    def test_line_number_replacement(self):
        msg = "Error at line 42 in module"
        result = normalize_message(msg)
        assert "line <N>" in result
        assert "42" not in result

    def test_quoted_identifier_single(self):
        msg = "cannot import name 'foo_bar' from 'baz.qux'"
        result = normalize_message(msg)
        assert "'<ID>'" in result
        assert "foo_bar" not in result

    def test_quoted_identifier_double(self):
        msg = 'module "my_module" has no attribute "thing"'
        result = normalize_message(msg)
        assert '"<ID>"' in result
        assert "my_module" not in result

    def test_uuid_replacement(self):
        msg = "Failed to process entity 550e8400-e29b-41d4-a716-446655440000"
        result = normalize_message(msg)
        assert "<UUID>" in result
        assert "550e8400" not in result

    def test_ip_replacement(self):
        msg = "Connection refused to 192.168.1.100:5432"
        result = normalize_message(msg)
        assert "<IP>" in result
        assert "192.168" not in result

    def test_url_replacement(self):
        msg = "Failed to fetch https://api.example.com/v2/users"
        result = normalize_message(msg)
        assert "<URL>" in result
        assert "example.com" not in result

    def test_hex_address_replacement(self):
        msg = "Segfault at address 0x7fff5fbff8c0"
        result = normalize_message(msg)
        assert "<HEX>" in result
        assert "0x7fff" not in result

    def test_large_number_replacement(self):
        msg = "Timeout after 30000ms waiting for element"
        result = normalize_message(msg)
        assert "<NUM>" in result
        assert "30000" not in result

    def test_whitespace_normalization(self):
        msg = "Error   with    extra   spaces"
        result = normalize_message(msg)
        assert "  " not in result

    def test_combined_normalization(self):
        msg = "ImportError: cannot import name 'foo' from 'bar' (/home/user/bar.py)"
        result = normalize_message(msg)
        assert "'<ID>'" in result
        assert "<PATH>" in result
        assert "foo" not in result

    def test_preserves_error_structure(self):
        msg = "TypeError: unsupported operand type"
        result = normalize_message(msg)
        # The core error structure should remain
        assert "TypeError:" in result or "unsupported operand type" in result


# =========================================================================
# extract_frames
# =========================================================================


class TestExtractFrames:
    """Test stack frame extraction and stdlib filtering."""

    def test_python_frames(self):
        trace = (
            "Traceback (most recent call last):\n"
            '  File "/home/user/app/main.py", line 10, in run_app\n'
            "    result = process_data(input)\n"
            '  File "/home/user/app/processor.py", line 25, in process_data\n'
            "    return validate(data)\n"
            '  File "/home/user/app/validator.py", line 8, in validate\n'
            '    raise ValueError("bad")\n'
        )
        frames = extract_frames(trace)
        assert frames == ["run_app", "process_data", "validate"]

    def test_python_filters_stdlib(self):
        trace = (
            "Traceback (most recent call last):\n"
            '  File "/usr/lib/python3.11/importlib/__init__.py", line 126, in import_module\n'
            "    return _bootstrap._gcd_import(name)\n"
            '  File "/home/user/app/main.py", line 5, in load_plugins\n'
            "    import plugin\n"
        )
        frames = extract_frames(trace)
        assert frames == ["load_plugins"]
        assert "import_module" not in frames

    def test_python_filters_site_packages(self):
        trace = (
            "Traceback (most recent call last):\n"
            '  File "/home/user/.venv/lib/python3.11/site-packages/flask/app.py", line 100, in dispatch\n'
            "    return self.handle()\n"
            '  File "/home/user/app/views.py", line 42, in handle_request\n'
            "    return render(data)\n"
        )
        frames = extract_frames(trace)
        assert frames == ["handle_request"]

    def test_js_frames(self):
        trace = (
            "Error: connection failed\n"
            "    at connectDB (/home/user/app/db.js:15:10)\n"
            "    at initServer (/home/user/app/server.js:42:5)\n"
            "    at main (/home/user/app/index.js:8:3)\n"
        )
        frames = extract_frames(trace)
        assert frames == ["connectDB", "initServer", "main"]

    def test_js_filters_node_modules(self):
        trace = (
            "Error: timeout\n"
            "    at Timeout._onTimeout (node_modules/express/lib/router.js:100:5)\n"
            "    at handleRequest (/home/user/app/handler.js:20:10)\n"
        )
        frames = extract_frames(trace)
        assert frames == ["handleRequest"]

    def test_js_anonymous_frames(self):
        trace = (
            "Error: failed\n"
            "    at /home/user/app/script.js:10:5\n"
            "    at processData (/home/user/app/data.js:25:3)\n"
        )
        frames = extract_frames(trace)
        assert "<anonymous>" in frames
        assert "processData" in frames

    def test_empty_trace(self):
        assert extract_frames("") == []

    def test_no_trace(self):
        assert extract_frames("just an error message, no trace") == []

    def test_all_stdlib_frames_returns_empty(self):
        trace = (
            "Traceback (most recent call last):\n"
            '  File "/usr/lib/python3.11/threading.py", line 1038, in run\n'
            "    self._target()\n"
            '  File "/usr/lib/python3.11/concurrent/futures/thread.py", line 58, in run\n'
            "    result = self.fn(*self.args)\n"
        )
        frames = extract_frames(trace)
        assert frames == []


# =========================================================================
# compute_fingerprint — stability and differentiation
# =========================================================================


class TestComputeFingerprint:
    """Test fingerprint generation, stability, and differentiation."""

    def test_same_error_same_fingerprint(self):
        """Identical errors must produce identical fingerprints."""
        fp1 = compute_fingerprint("TypeError: bad operand", "", "logic")
        fp2 = compute_fingerprint("TypeError: bad operand", "", "logic")
        assert fp1 == fp2

    def test_fingerprint_is_64_hex_chars(self):
        """SHA-256 produces a 64-character hex string."""
        fp = compute_fingerprint("ImportError: no module named 'foo'")
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_different_paths_same_fingerprint(self):
        """Same error type in different files should match."""
        fp1 = compute_fingerprint(
            "ImportError: cannot import name 'handler' from 'app.views' (/home/alice/project/app/views.py)"
        )
        fp2 = compute_fingerprint(
            "ImportError: cannot import name 'handler' from 'app.views' (/home/bob/project/app/views.py)"
        )
        assert fp1 == fp2

    def test_different_line_numbers_same_fingerprint(self):
        """Same error at different line numbers should match."""
        trace1 = '  File "/app/main.py", line 10, in run\n    foo()\n'
        trace2 = '  File "/app/main.py", line 99, in run\n    foo()\n'
        fp1 = compute_fingerprint("RuntimeError: boom", trace1)
        fp2 = compute_fingerprint("RuntimeError: boom", trace2)
        assert fp1 == fp2

    def test_different_variable_names_same_fingerprint(self):
        """Same error pattern with different variable names should match."""
        fp1 = compute_fingerprint("NameError: name 'xyz' is not defined")
        fp2 = compute_fingerprint("NameError: name 'abc' is not defined")
        assert fp1 == fp2

    def test_different_error_types_different_fingerprint(self):
        """Different exception types should produce different fingerprints."""
        fp1 = compute_fingerprint("TypeError: bad value")
        fp2 = compute_fingerprint("ValueError: bad value")
        assert fp1 != fp2

    def test_different_categories_different_fingerprint(self):
        """Different error categories should produce different fingerprints."""
        fp1 = compute_fingerprint("Error: timeout", "", "performance")
        fp2 = compute_fingerprint("Error: timeout", "", "integration")
        assert fp1 != fp2

    def test_different_call_paths_different_fingerprint(self):
        """Different call paths should produce different fingerprints."""
        trace1 = '  File "/app/a.py", line 1, in func_a\n    raise ValueError("x")\n'
        trace2 = '  File "/app/b.py", line 1, in func_b\n    raise ValueError("x")\n'
        fp1 = compute_fingerprint("ValueError: x", trace1)
        fp2 = compute_fingerprint("ValueError: x", trace2)
        assert fp1 != fp2

    def test_with_and_without_stack_trace(self):
        """Adding a stack trace should change the fingerprint when it has app frames."""
        trace = '  File "/app/main.py", line 10, in run\n    foo()\n'
        fp_no_trace = compute_fingerprint("RuntimeError: fail")
        fp_with_trace = compute_fingerprint("RuntimeError: fail", trace)
        assert fp_no_trace != fp_with_trace

    def test_empty_trace_same_as_no_trace(self):
        """Empty string trace should equal no trace."""
        fp1 = compute_fingerprint("TypeError: boom", "")
        fp2 = compute_fingerprint("TypeError: boom")
        assert fp1 == fp2

    def test_stdlib_only_trace_same_as_no_trace(self):
        """Stack trace with only stdlib frames should match no-trace fingerprint."""
        stdlib_trace = (
            '  File "/usr/lib/python3.11/threading.py", line 100, in run\n'
            "    self._target()\n"
        )
        fp1 = compute_fingerprint("RuntimeError: fail")
        fp2 = compute_fingerprint("RuntimeError: fail", stdlib_trace)
        assert fp1 == fp2


# =========================================================================
# analyze_error — full pipeline
# =========================================================================


class TestAnalyzeError:
    """Test the full analyze_error() pipeline."""

    def test_returns_expected_keys(self):
        result = analyze_error(
            "TypeError: bad", '  File "/app/main.py", line 1, in run\n', "logic"
        )
        assert "fingerprint" in result
        assert "exception_type" in result
        assert "message_template" in result
        assert "app_frames" in result
        assert "error_category" in result
        assert "frame_count" in result

    def test_exception_type_populated(self):
        result = analyze_error("ImportError: no module 'foo'")
        assert result["exception_type"] == "ImportError"

    def test_message_template_is_normalized(self):
        result = analyze_error("ImportError: cannot import name 'foo' from 'bar'")
        assert "'<ID>'" in result["message_template"]

    def test_app_frames_populated(self):
        trace = '  File "/app/main.py", line 10, in run_app\n    do_thing()\n'
        result = analyze_error("RuntimeError: boom", trace)
        assert result["app_frames"] == ["run_app"]
        assert result["frame_count"] == 1

    def test_category_passed_through(self):
        result = analyze_error("Error: slow query", error_category="performance")
        assert result["error_category"] == "performance"

    def test_fingerprint_matches_compute(self):
        """analyze_error fingerprint should match compute_fingerprint."""
        msg = "TypeError: cannot add int and str"
        trace = '  File "/app/math.py", line 5, in add\n    return a + b\n'
        result = analyze_error(msg, trace, "logic")
        expected = compute_fingerprint(msg, trace, "logic")
        assert result["fingerprint"] == expected

    def test_unknown_error_type(self):
        result = analyze_error("Something broke completely")
        assert result["exception_type"] == "UnknownError"

    def test_no_frames_zero_count(self):
        result = analyze_error("TypeError: bad")
        assert result["app_frames"] == []
        assert result["frame_count"] == 0
