"""
engram.signal.style -- Architectural style adherence and pattern drift detection.

Replaces the old consciousness/identity.py dissociation detector.
Where that module used regex to detect phrases like "as an AI" (identity
drift), this module uses regex to detect code style violations (pattern
drift) — naming inconsistencies, forbidden constructs, and anti-patterns.

The drift detector is configurable per-project via StyleProfile, which
defines the expected naming convention, forbidden patterns, and
required patterns for generated code.

Usage:
    profile = StyleProfile.python_default()
    result = assess_style(code_text, profile)
    # result.score -> 0.0 to 1.0
    # result.violations -> list of detected violations
    # result.state -> "consistent" | "minor_drift" | "significant_drift" | "incompatible"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Violation
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    """A single detected style violation."""

    category: str  # "naming", "forbidden", "missing_required", "anti_pattern"
    description: str  # Human-readable description
    severity: float  # 0.0 (trivial) to 1.0 (critical)
    line_hint: str = ""  # Optional: the offending text snippet


# ---------------------------------------------------------------------------
# StyleResult
# ---------------------------------------------------------------------------


@dataclass
class StyleResult:
    """Assessment result from style analysis."""

    score: float  # 0.0 (many violations) to 1.0 (fully consistent)
    violations: List[Violation] = field(default_factory=list)
    patterns_checked: int = 0  # Total number of pattern checks performed

    @property
    def state(self) -> str:
        """Categorical state label for the style assessment."""
        if self.score >= 0.90:
            return "consistent"
        elif self.score >= 0.70:
            return "minor_drift"
        elif self.score >= 0.40:
            return "significant_drift"
        else:
            return "incompatible"

    @property
    def needs_correction(self) -> bool:
        return self.score < 0.70

    def to_dict(self) -> Dict:
        return {
            "score": round(self.score, 3),
            "state": self.state,
            "needs_correction": self.needs_correction,
            "violations": [
                {
                    "category": v.category,
                    "description": v.description,
                    "severity": v.severity,
                    "line_hint": v.line_hint,
                }
                for v in self.violations
            ],
            "patterns_checked": self.patterns_checked,
        }


# ---------------------------------------------------------------------------
# StyleProfile
# ---------------------------------------------------------------------------


@dataclass
class StyleProfile:
    """Project-specific style configuration.

    Defines naming conventions, forbidden constructs, required patterns,
    and anti-patterns to detect in generated code.  Each project can
    have a different profile.
    """

    # Naming convention: "snake_case", "camelCase", "PascalCase", "any"
    naming_convention: str = "snake_case"

    # Forbidden patterns: regex -> description.
    # Code matching any of these is flagged as a violation.
    forbidden_patterns: Dict[str, str] = field(default_factory=dict)

    # Required patterns: regex -> description.
    # If the code LACKS any of these, it's flagged as a violation.
    # Only checked if the code is long enough (>5 lines).
    required_patterns: Dict[str, str] = field(default_factory=dict)

    # Anti-patterns: regex -> (description, severity).
    # Less severe than forbidden — these are warnings not errors.
    anti_patterns: Dict[str, Tuple[str, float]] = field(default_factory=dict)

    # Maximum allowed nesting depth (0 = don't check)
    max_nesting_depth: int = 0

    # Maximum allowed line length (0 = don't check)
    max_line_length: int = 0

    @classmethod
    def python_default(cls) -> "StyleProfile":
        """Default style profile for Python projects."""
        return cls(
            naming_convention="snake_case",
            forbidden_patterns={
                r"\beval\s*\(": "eval() is forbidden — use ast.literal_eval() or explicit parsing",
                r"\bexec\s*\(": "exec() is forbidden — use explicit function dispatch",
                r"from \S+ import \*": "Wildcard imports are forbidden — import specific names",
                r"\bprint\s*\((?!.*#\s*debug)": "print() without #debug comment — use logging module",
                r"except\s*:": "Bare except clause — catch specific exceptions",
                r"except\s+Exception\s*:": "Broad except Exception — catch specific exceptions",
            },
            required_patterns={
                r'(?:"""|\'\'\'|#\s*\w)': "Missing docstrings/comments — add documentation",
            },
            anti_patterns={
                r"# ?TODO|# ?FIXME|# ?HACK|# ?XXX": (
                    "TODO/FIXME/HACK comment found — track or resolve",
                    0.3,
                ),
                r"type:\s*ignore": (
                    "type: ignore comment — fix the type issue instead",
                    0.4,
                ),
                r"noqa": (
                    "noqa comment — fix the linting issue instead",
                    0.3,
                ),
            },
            max_nesting_depth=4,
            max_line_length=120,
        )

    @classmethod
    def javascript_default(cls) -> "StyleProfile":
        """Default style profile for JavaScript/TypeScript projects."""
        return cls(
            naming_convention="camelCase",
            forbidden_patterns={
                r"\beval\s*\(": "eval() is forbidden — security risk",
                r"\bvar\s+": "var keyword — use let or const",
                r"==(?!=)": "Loose equality (==) — use strict equality (===)",
                r"!=(?!=)": "Loose inequality (!=) — use strict inequality (!==)",
                r"require\(": "require() in ES module project — use import",
            },
            required_patterns={},
            anti_patterns={
                r"any(?:\s|;|,|\))": (
                    "TypeScript 'any' type — use specific types",
                    0.4,
                ),
                r"// ?TODO|// ?FIXME|// ?HACK": (
                    "TODO/FIXME/HACK comment found — track or resolve",
                    0.3,
                ),
                r"@ts-ignore|@ts-expect-error": (
                    "TypeScript suppression — fix the type issue",
                    0.5,
                ),
            },
            max_nesting_depth=4,
            max_line_length=100,
        )

    @classmethod
    def minimal(cls) -> "StyleProfile":
        """Minimal profile — only the most critical checks."""
        return cls(
            naming_convention="any",
            forbidden_patterns={
                r"\beval\s*\(": "eval() is a security risk",
                r"except\s*:": "Bare except clause",
            },
            required_patterns={},
            anti_patterns={},
            max_nesting_depth=0,
            max_line_length=0,
        )


# ---------------------------------------------------------------------------
# Naming convention detection
# ---------------------------------------------------------------------------

# Regex patterns for identifying naming styles in identifiers
_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_CAMEL_CASE = re.compile(r"^[a-z][a-zA-Z0-9]*$")
_PASCAL_CASE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")
_UPPER_SNAKE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")

# Extract identifiers from code: function/method names, variable assignments
_FUNC_DEF = re.compile(r"(?:def|function|const|let|var)\s+([a-zA-Z_]\w*)")
_CLASS_DEF = re.compile(r"class\s+([a-zA-Z_]\w*)")
_ASSIGN = re.compile(r"^(\s*)([a-zA-Z_]\w*)\s*(?::\s*\w+)?\s*=", re.MULTILINE)


def _check_naming(
    code: str,
    convention: str,
) -> List[Violation]:
    """Check identifiers against the expected naming convention.

    Only checks function/method definitions and top-level variable
    assignments — not class names (which are always PascalCase by
    convention in most languages).
    """
    if convention == "any":
        return []

    checker = {
        "snake_case": _SNAKE_CASE,
        "camelCase": _CAMEL_CASE,
        "PascalCase": _PASCAL_CASE,
    }.get(convention)

    if checker is None:
        return []

    violations: List[Violation] = []

    # Check function/method definitions
    for match in _FUNC_DEF.finditer(code):
        name = match.group(1)
        # Skip dunder methods, single-char names, and ALL_CAPS constants
        if name.startswith("_") or len(name) <= 1 or _UPPER_SNAKE.match(name):
            continue
        if not checker.match(name):
            violations.append(
                Violation(
                    category="naming",
                    description=f"'{name}' doesn't follow {convention} convention",
                    severity=0.3,
                    line_hint=name,
                )
            )

    # Check variable assignments (only non-indented or single-indent)
    for match in _ASSIGN.finditer(code):
        indent = match.group(1)
        name = match.group(2)
        # Skip deeply indented (loop vars etc), dunder, short names, constants
        if len(indent) > 4 or name.startswith("_") or len(name) <= 1:
            continue
        if _UPPER_SNAKE.match(name):
            continue  # Constants are always UPPER_SNAKE
        if not checker.match(name):
            violations.append(
                Violation(
                    category="naming",
                    description=f"'{name}' doesn't follow {convention} convention",
                    severity=0.2,
                    line_hint=name,
                )
            )

    return violations


# ---------------------------------------------------------------------------
# Nesting depth check
# ---------------------------------------------------------------------------


def _check_nesting_depth(code: str, max_depth: int) -> List[Violation]:
    """Check for excessive nesting depth in code."""
    if max_depth <= 0:
        return []

    violations: List[Violation] = []
    lines = code.split("\n")

    for i, line in enumerate(lines):
        if not line.strip():
            continue
        # Estimate indent level (4 spaces = 1 level, or 1 tab = 1 level)
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        # Normalize: tabs count as 4 spaces
        indent = indent if "\t" not in line[:indent] else line[:indent].count("\t") * 4
        level = indent // 4

        if level > max_depth and stripped and not stripped.startswith(("#", "//", "*")):
            violations.append(
                Violation(
                    category="anti_pattern",
                    description=f"Nesting depth {level} exceeds maximum {max_depth}",
                    severity=0.4,
                    line_hint=stripped[:80],
                )
            )
            break  # One violation is enough

    return violations


# ---------------------------------------------------------------------------
# Line length check
# ---------------------------------------------------------------------------


def _check_line_length(code: str, max_length: int) -> List[Violation]:
    """Check for lines exceeding max length."""
    if max_length <= 0:
        return []

    violations: List[Violation] = []
    long_count = 0

    for line in code.split("\n"):
        if len(line) > max_length:
            long_count += 1

    if long_count > 0:
        severity = min(0.5, 0.1 * long_count)
        violations.append(
            Violation(
                category="anti_pattern",
                description=f"{long_count} line(s) exceed {max_length} chars",
                severity=severity,
                line_hint=f"{long_count} lines",
            )
        )

    return violations


# ---------------------------------------------------------------------------
# Main assessment function
# ---------------------------------------------------------------------------


def assess_style(
    code: str,
    profile: Optional[StyleProfile] = None,
) -> StyleResult:
    """Assess code against a style profile and return a StyleResult.

    Parameters
    ----------
    code:
        The code text to assess.
    profile:
        The style profile to check against.  Defaults to
        ``StyleProfile.python_default()`` if not provided.

    Returns
    -------
    StyleResult
        Score (0-1), violation list, and categorical state.
    """
    if profile is None:
        profile = StyleProfile.python_default()

    if not code or not code.strip():
        return StyleResult(score=1.0, patterns_checked=0)

    violations: List[Violation] = []
    patterns_checked = 0

    # 1. Naming convention check
    naming_violations = _check_naming(code, profile.naming_convention)
    violations.extend(naming_violations)
    patterns_checked += 1

    # 2. Forbidden patterns
    for pattern_str, description in profile.forbidden_patterns.items():
        patterns_checked += 1
        try:
            pattern = re.compile(pattern_str)
            matches = pattern.findall(code)
            if matches:
                violations.append(
                    Violation(
                        category="forbidden",
                        description=description,
                        severity=0.7,
                        line_hint=matches[0]
                        if isinstance(matches[0], str)
                        else str(matches[0]),
                    )
                )
        except re.error:
            pass  # Skip invalid regex

    # 3. Required patterns (only for substantial code blocks)
    lines = [l for l in code.split("\n") if l.strip()]
    if len(lines) > 5:
        for pattern_str, description in profile.required_patterns.items():
            patterns_checked += 1
            try:
                pattern = re.compile(pattern_str)
                if not pattern.search(code):
                    violations.append(
                        Violation(
                            category="missing_required",
                            description=description,
                            severity=0.4,
                            line_hint="",
                        )
                    )
            except re.error:
                pass

    # 4. Anti-patterns
    for pattern_str, (description, severity) in profile.anti_patterns.items():
        patterns_checked += 1
        try:
            pattern = re.compile(pattern_str)
            matches = pattern.findall(code)
            if matches:
                violations.append(
                    Violation(
                        category="anti_pattern",
                        description=description,
                        severity=severity,
                        line_hint=matches[0]
                        if isinstance(matches[0], str)
                        else str(matches[0]),
                    )
                )
        except re.error:
            pass

    # 5. Nesting depth
    patterns_checked += 1
    nesting_violations = _check_nesting_depth(code, profile.max_nesting_depth)
    violations.extend(nesting_violations)

    # 6. Line length
    patterns_checked += 1
    line_violations = _check_line_length(code, profile.max_line_length)
    violations.extend(line_violations)

    # -- Score calculation ------------------------------------------------
    # Each violation reduces the score proportionally to its severity.
    # Maximum penalty is 1.0 (score floor at 0.0).
    if not violations:
        score = 1.0
    else:
        total_penalty = sum(v.severity for v in violations)
        # Diminishing returns: first violations hurt more
        score = max(0.0, 1.0 - (total_penalty / (total_penalty + 2.0)))

    return StyleResult(
        score=score,
        violations=violations,
        patterns_checked=patterns_checked,
    )


def build_style_correction(result: StyleResult) -> str:
    """Build a correction prompt from style assessment results.

    Returns an empty string if the code is consistent (no correction needed).
    """
    if not result.needs_correction:
        return ""

    lines = [
        f"[Style drift detected: score {result.score:.2f} — {result.state}]",
        "The following style issues were found:",
    ]

    # Group by category
    by_category: Dict[str, List[Violation]] = {}
    for v in result.violations:
        by_category.setdefault(v.category, []).append(v)

    for category, viols in by_category.items():
        label = {
            "naming": "Naming Convention",
            "forbidden": "Forbidden Construct",
            "missing_required": "Missing Requirement",
            "anti_pattern": "Anti-Pattern",
        }.get(category, category.title())

        for v in viols[:3]:  # Cap at 3 per category
            lines.append(f"  - [{label}] {v.description}")

    lines.append("Fix these issues in your next response.")
    return "\n".join(lines)
