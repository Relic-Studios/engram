"""Tests for the architectural style adherence / pattern drift detector.

Replaces the old consciousness/identity.py dissociation tests with
code-focused style drift detection.
"""

import pytest
from engram.signal.style import (
    assess_style,
    build_style_correction,
    StyleProfile,
    StyleResult,
    Violation,
    _check_naming,
    _check_nesting_depth,
    _check_line_length,
)


# ---------------------------------------------------------------------------
# StyleProfile tests
# ---------------------------------------------------------------------------


class TestStyleProfile:
    """Test profile construction and presets."""

    def test_python_default(self):
        profile = StyleProfile.python_default()
        assert profile.naming_convention == "snake_case"
        assert len(profile.forbidden_patterns) > 0
        assert profile.max_nesting_depth == 4

    def test_javascript_default(self):
        profile = StyleProfile.javascript_default()
        assert profile.naming_convention == "camelCase"
        assert r"\bvar\s+" in profile.forbidden_patterns

    def test_minimal(self):
        profile = StyleProfile.minimal()
        assert profile.naming_convention == "any"
        assert profile.max_nesting_depth == 0

    def test_custom_profile(self):
        profile = StyleProfile(
            naming_convention="PascalCase",
            forbidden_patterns={r"goto": "No goto statements"},
        )
        assert profile.naming_convention == "PascalCase"
        assert len(profile.forbidden_patterns) == 1


# ---------------------------------------------------------------------------
# Naming convention tests
# ---------------------------------------------------------------------------


class TestNamingCheck:
    """Test identifier naming convention detection."""

    def test_snake_case_correct(self):
        code = "def process_user_data(user_id):\n    pass"
        violations = _check_naming(code, "snake_case")
        assert len(violations) == 0

    def test_camel_case_in_snake_project(self):
        code = "def processUserData(userId):\n    pass"
        violations = _check_naming(code, "snake_case")
        assert len(violations) > 0
        assert any("processUserData" in v.line_hint for v in violations)

    def test_snake_case_in_camel_project(self):
        code = "def process_user_data(user_id):\n    pass"
        violations = _check_naming(code, "camelCase")
        assert len(violations) > 0

    def test_any_convention_skips_check(self):
        code = "def processUserData(userId):\n    pass"
        violations = _check_naming(code, "any")
        assert len(violations) == 0

    def test_dunder_methods_skipped(self):
        code = "def __init__(self):\n    pass"
        violations = _check_naming(code, "snake_case")
        assert len(violations) == 0

    def test_constants_skipped(self):
        code = "MAX_RETRIES = 3\nDEFAULT_TIMEOUT = 30"
        violations = _check_naming(code, "snake_case")
        assert len(violations) == 0

    def test_single_char_skipped(self):
        code = "def f():\n    x = 1"
        violations = _check_naming(code, "snake_case")
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Nesting depth tests
# ---------------------------------------------------------------------------


class TestNestingDepth:
    """Test excessive nesting detection."""

    def test_acceptable_nesting(self):
        code = "def foo():\n    if True:\n        for x in y:\n            pass"
        violations = _check_nesting_depth(code, max_depth=4)
        assert len(violations) == 0

    def test_excessive_nesting(self):
        # 6 levels of indentation (24 spaces)
        code = "def foo():\n    if True:\n        for x in y:\n            if z:\n                for a in b:\n                        deeply_nested()"
        violations = _check_nesting_depth(code, max_depth=4)
        assert len(violations) > 0

    def test_disabled_check(self):
        code = "                        deeply_nested()"
        violations = _check_nesting_depth(code, max_depth=0)
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Line length tests
# ---------------------------------------------------------------------------


class TestLineLength:
    """Test line length violation detection."""

    def test_acceptable_length(self):
        code = "x = 1\ny = 2\nresult = x + y"
        violations = _check_line_length(code, max_length=120)
        assert len(violations) == 0

    def test_long_lines(self):
        code = "x = " + "a" * 200 + "\ny = " + "b" * 200
        violations = _check_line_length(code, max_length=120)
        assert len(violations) > 0
        assert "2 line(s)" in violations[0].description

    def test_disabled_check(self):
        code = "x = " + "a" * 200
        violations = _check_line_length(code, max_length=0)
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Full assessment tests
# ---------------------------------------------------------------------------


class TestAssessStyle:
    """Integration tests for the full style assessment."""

    def test_clean_code_gets_high_score(self):
        code = '''def process_items(items):
    """Process a list of items and return results."""
    results = []
    for item in items:
        try:
            result = validate_item(item)
            results.append(result)
        except ValueError as exc:
            log.warning("Invalid item: %s", exc)
    return results
'''
        result = assess_style(code, StyleProfile.python_default())
        assert result.score >= 0.7
        assert result.state in ("consistent", "minor_drift")

    def test_forbidden_eval_detected(self):
        code = "result = eval(user_input)"
        result = assess_style(code, StyleProfile.python_default())
        assert any(v.category == "forbidden" for v in result.violations)
        assert any("eval" in v.description for v in result.violations)

    def test_bare_except_detected(self):
        code = "try:\n    foo()\nexcept:\n    pass"
        result = assess_style(code, StyleProfile.python_default())
        assert any("Bare except" in v.description for v in result.violations)

    def test_wildcard_import_detected(self):
        code = "from os.path import *\ndef foo():\n    pass"
        result = assess_style(code, StyleProfile.python_default())
        assert any("Wildcard" in v.description for v in result.violations)

    def test_empty_code_perfect_score(self):
        result = assess_style("", StyleProfile.python_default())
        assert result.score == 1.0
        assert result.state == "consistent"

    def test_none_profile_uses_python_default(self):
        code = "def foo():\n    pass"
        result = assess_style(code, None)
        assert isinstance(result, StyleResult)

    def test_many_violations_low_score(self):
        code = """from os import *
result = eval(user_input)
exec(command)
try:
    dangerous()
except:
    pass
"""
        result = assess_style(code, StyleProfile.python_default())
        assert result.score < 0.5
        assert result.state in ("significant_drift", "incompatible")

    def test_javascript_var_detected(self):
        code = "var x = 1;\nvar y = 2;"
        result = assess_style(code, StyleProfile.javascript_default())
        assert any("var" in v.description.lower() for v in result.violations)


# ---------------------------------------------------------------------------
# StyleResult tests
# ---------------------------------------------------------------------------


class TestStyleResult:
    """Test StyleResult properties and serialization."""

    def test_state_consistent(self):
        result = StyleResult(score=0.95)
        assert result.state == "consistent"
        assert not result.needs_correction

    def test_state_minor_drift(self):
        result = StyleResult(score=0.75)
        assert result.state == "minor_drift"
        assert not result.needs_correction

    def test_state_significant_drift(self):
        result = StyleResult(score=0.55)
        assert result.state == "significant_drift"
        assert result.needs_correction

    def test_state_incompatible(self):
        result = StyleResult(score=0.2)
        assert result.state == "incompatible"
        assert result.needs_correction

    def test_to_dict(self):
        result = StyleResult(
            score=0.8,
            violations=[Violation("naming", "bad name", 0.3, "processData")],
            patterns_checked=5,
        )
        d = result.to_dict()
        assert d["score"] == 0.8
        assert d["state"] == "minor_drift"
        assert len(d["violations"]) == 1
        assert d["violations"][0]["category"] == "naming"


# ---------------------------------------------------------------------------
# Correction prompt tests
# ---------------------------------------------------------------------------


class TestBuildStyleCorrection:
    """Test correction prompt generation."""

    def test_no_correction_when_consistent(self):
        result = StyleResult(score=0.9)
        assert build_style_correction(result) == ""

    def test_correction_when_drifted(self):
        result = StyleResult(
            score=0.4,
            violations=[
                Violation("forbidden", "eval() is dangerous", 0.7, "eval("),
                Violation("naming", "bad name", 0.3, "processData"),
            ],
        )
        correction = build_style_correction(result)
        assert "Style drift detected" in correction
        assert "eval" in correction
        assert "bad name" in correction

    def test_correction_caps_per_category(self):
        """At most 3 violations per category in correction prompt."""
        violations = [
            Violation("naming", f"bad name {i}", 0.2, f"name_{i}") for i in range(10)
        ]
        result = StyleResult(score=0.3, violations=violations)
        correction = build_style_correction(result)
        # Should only have 3 naming entries
        assert correction.count("Naming Convention") <= 3
