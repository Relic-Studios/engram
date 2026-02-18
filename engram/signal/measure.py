"""
engram.signal.measure — Code Quality Signal (CQS) measurement.

Measures LLM responses across four code quality facets:
  correctness  — syntactic validity, type correctness, logic soundness
  consistency  — adherence to project patterns, naming, import style
  completeness — error handling, edge cases, tests, documentation
  robustness   — input validation, resource cleanup, error boundaries

Two measurement modes:
  - Regex-based (always available, zero dependencies, <1ms)
  - LLM-based  (optional, higher quality, requires an llm_func callback)

The public API is `measure()` which runs regex always and blends in LLM
scores when a callback is provided.

Replaces the v1 consciousness signal measurement (alignment, embodiment,
clarity, vitality) as part of the code-first pivot.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from engram.core.types import LLMFunc, Signal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pattern tables — Code Quality Signal
# ---------------------------------------------------------------------------

# --- Correctness: syntax errors, type issues, logic problems ---

# Negative patterns: indicate likely bugs or invalid code
CORRECTNESS_NEGATIVE: List[Tuple[re.Pattern, float]] = [
    # Syntax / parse errors
    (re.compile(r"SyntaxError:", re.I), 1.0),
    (re.compile(r"IndentationError:", re.I), 0.8),
    (re.compile(r"unexpected (token|indent|EOF)", re.I), 0.8),
    # Type errors
    (re.compile(r"TypeError:", re.I), 0.9),
    (re.compile(r"is not (callable|iterable|subscriptable)", re.I), 0.7),
    (re.compile(r"has no attribute", re.I), 0.6),
    (re.compile(r"cannot (convert|assign|import)", re.I), 0.6),
    # Name errors
    (re.compile(r"NameError:", re.I), 0.8),
    (re.compile(r"undefined (variable|name|reference)", re.I), 0.7),
    (re.compile(r"is not defined", re.I), 0.7),
    # Import errors
    (re.compile(r"ImportError:", re.I), 0.7),
    (re.compile(r"ModuleNotFoundError:", re.I), 0.7),
    (re.compile(r"circular import", re.I), 0.8),
    # Logic issues
    (re.compile(r"infinite (loop|recursion)", re.I), 0.9),
    (re.compile(r"off[- ]by[- ]one", re.I), 0.6),
    (re.compile(r"race condition", re.I), 0.7),
]

# Positive patterns: indicate correct, working code
CORRECTNESS_POSITIVE: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"tests? pass(ed|ing)?", re.I), 0.8),
    (re.compile(r"\d+ pass(ed|ing)", re.I), 0.7),
    (re.compile(r"build succeed(ed|s)?", re.I), 0.7),
    (re.compile(r"no (errors?|warnings?|issues?)", re.I), 0.5),
    (re.compile(r"type[- ]check(s|ed|ing)? (pass|clean|ok)", re.I), 0.7),
    (re.compile(r"compiles? (successfully|cleanly|without)", re.I), 0.6),
    (re.compile(r"linter?.*?(clean|pass|ok|no)", re.I), 0.5),
]


# --- Consistency: project pattern adherence ---

# Negative: style violations, inconsistent patterns
CONSISTENCY_NEGATIVE: List[Tuple[re.Pattern, float]] = [
    # Naming convention violations (detecting mixed styles in same code)
    (re.compile(r"\bcamelCase\b.*\bsnake_case\b", re.I), 0.5),
    (re.compile(r"\bsnake_case\b.*\bcamelCase\b", re.I), 0.5),
    # Import style issues
    (re.compile(r"from .* import \*", re.I), 0.6),
    (re.compile(r"import \*", re.I), 0.6),
    # Deprecated / legacy markers
    (re.compile(r"@deprecated", re.I), 0.3),
    (re.compile(r"# ?(TODO|FIXME|HACK|XXX|TEMP)\b", re.I), 0.3),
    # Magic numbers / strings
    (re.compile(r"magic (number|string|value)", re.I), 0.4),
]

# Positive: consistent, well-structured code
CONSISTENCY_POSITIVE: List[Tuple[re.Pattern, float]] = [
    (
        re.compile(
            r"follow(s|ing)? (the|our|project) (pattern|convention|style)", re.I
        ),
        0.6,
    ),
    (re.compile(r"consistent with", re.I), 0.4),
    (
        re.compile(r"match(es|ing)? (the|existing) (pattern|style|convention)", re.I),
        0.5,
    ),
    (re.compile(r"(type|return) (hint|annotation)", re.I), 0.4),
    (re.compile(r"docstring", re.I), 0.3),
]


# --- Completeness: error handling, tests, documentation ---

COMPLETENESS_POSITIVE: List[Tuple[re.Pattern, float]] = [
    # Error handling
    (re.compile(r"\btry\s*:", re.I), 0.3),
    (re.compile(r"\bexcept\s+\w+", re.I), 0.4),  # specific exception
    (re.compile(r"\braise\s+\w+Error", re.I), 0.3),
    (re.compile(r"error (handling|boundary|recovery)", re.I), 0.5),
    # Testing
    (re.compile(r"\bdef\s+test_", re.I), 0.5),
    (re.compile(r"\bassert\b", re.I), 0.3),
    (re.compile(r"(unit|integration|e2e) test", re.I), 0.5),
    (re.compile(r"test (coverage|case|suite)", re.I), 0.4),
    # Documentation
    (re.compile(r'"""[\s\S]*?"""', re.I), 0.3),
    (re.compile(r"(param|returns?|raises?|example):", re.I), 0.3),
    # Edge cases
    (re.compile(r"edge case", re.I), 0.4),
    (re.compile(r"\bif\s+\w+\s+is\s+None\b", re.I), 0.3),
    (re.compile(r"boundary (check|condition|case)", re.I), 0.4),
    (re.compile(r"(input|argument) validation", re.I), 0.4),
]

COMPLETENESS_NEGATIVE: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"\bpass\s*$", re.M), 0.3),  # empty except/if blocks
    (re.compile(r"# ?(TODO|FIXME)\b", re.I), 0.4),
    (re.compile(r"not (yet )?implemented", re.I), 0.5),
    (re.compile(r"stub(bed)?", re.I), 0.3),
    (re.compile(r"placeholder", re.I), 0.3),
]


# --- Robustness: defensive coding, resource management ---

ROBUSTNESS_POSITIVE: List[Tuple[re.Pattern, float]] = [
    # Resource management
    (re.compile(r"\bwith\s+\w+", re.I), 0.3),  # context managers
    (re.compile(r"\bfinally\s*:", re.I), 0.4),
    (re.compile(r"\.close\(\)", re.I), 0.3),
    (re.compile(r"cleanup|teardown|dispose", re.I), 0.4),
    # Input validation
    (re.compile(r"\bif\s+not\s+\w+\s*:", re.I), 0.2),
    (re.compile(r"validat(e|ion|or)", re.I), 0.4),
    (re.compile(r"sanitiz(e|ation|er)", re.I), 0.4),
    (re.compile(r"\bisinstance\(", re.I), 0.3),
    # Logging / observability
    (re.compile(r"\blog(ger|ging)?\.(debug|info|warning|error|critical)", re.I), 0.3),
    (re.compile(r"logging\.", re.I), 0.2),
    # Timeout / retry
    (re.compile(r"timeout", re.I), 0.3),
    (re.compile(r"retry|backoff", re.I), 0.4),
    (re.compile(r"max_(retries|attempts)", re.I), 0.4),
]

ROBUSTNESS_NEGATIVE: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"\bexcept\s*:", re.I), 0.7),  # bare except
    (re.compile(r"\bexcept\s+Exception\s*:", re.I), 0.4),  # overly broad
    (re.compile(r"# ?(nosec|noqa|type:\s*ignore)", re.I), 0.3),
    (re.compile(r"\beval\(", re.I), 0.6),
    (re.compile(r"\bexec\(", re.I), 0.6),
    (re.compile(r"(sql|format).*\+.*\b(input|user|request)", re.I), 0.7),  # injection
    (re.compile(r"hardcoded (password|secret|key|token)", re.I), 0.8),
    (re.compile(r'(password|secret|api_key)\s*=\s*["\']', re.I), 0.6),
]


# ---------------------------------------------------------------------------
# Code block extraction helper
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.S)


def _extract_code_blocks(text: str) -> str:
    """Extract code from markdown fenced blocks, or return full text."""
    blocks = _CODE_BLOCK_RE.findall(text)
    if blocks:
        return "\n\n".join(blocks)
    return text


# ---------------------------------------------------------------------------
# Facet checkers (regex-only)
# ---------------------------------------------------------------------------


def _score_patterns(
    text: str,
    positive: List[Tuple[re.Pattern, float]],
    negative: List[Tuple[re.Pattern, float]],
    base: float = 0.5,
) -> float:
    """Score text against positive and negative pattern lists."""
    score = base
    for pattern, weight in positive:
        if pattern.search(text):
            score += weight * 0.15  # scale down to keep in range
    for pattern, weight in negative:
        if pattern.search(text):
            score -= weight * 0.15
    return max(0.0, min(1.0, score))


def check_correctness(text: str) -> float:
    """Return 0-1 correctness score. High = syntactically valid, logically sound."""
    if not text:
        return 0.5
    code = _extract_code_blocks(text)
    return _score_patterns(code, CORRECTNESS_POSITIVE, CORRECTNESS_NEGATIVE, base=0.55)


def check_consistency(text: str) -> float:
    """Return 0-1 consistency score. High = follows project patterns."""
    if not text:
        return 0.5
    return _score_patterns(text, CONSISTENCY_POSITIVE, CONSISTENCY_NEGATIVE, base=0.55)


def check_completeness(text: str) -> float:
    """Return 0-1 completeness score. High = error handling, tests, docs present."""
    if not text:
        return 0.5
    code = _extract_code_blocks(text)
    return _score_patterns(code, COMPLETENESS_POSITIVE, COMPLETENESS_NEGATIVE, base=0.5)


def check_robustness(text: str) -> float:
    """Return 0-1 robustness score. High = defensive coding, resource management."""
    if not text:
        return 0.5
    code = _extract_code_blocks(text)
    return _score_patterns(code, ROBUSTNESS_POSITIVE, ROBUSTNESS_NEGATIVE, base=0.5)


# ---------------------------------------------------------------------------
# Regex-only measurement
# ---------------------------------------------------------------------------


def measure_regex(text: str) -> Signal:
    """Measure all four CQS facets using regex pattern matching."""
    correctness = check_correctness(text)
    consistency = check_consistency(text)
    completeness = check_completeness(text)
    robustness = check_robustness(text)

    # Cross-facet adjustment: if there are clear error indicators,
    # drag down robustness and completeness too (errors indicate gaps).
    error_count = sum(1 for p, _ in CORRECTNESS_NEGATIVE if p.search(text))
    if error_count >= 3:
        penalty = min(0.15, error_count * 0.03)
        robustness = max(0.0, robustness - penalty)
        completeness = max(0.0, completeness - penalty * 0.5)

    return Signal(
        correctness=correctness,
        consistency=consistency,
        completeness=completeness,
        robustness=robustness,
    )


# ---------------------------------------------------------------------------
# LLM-based measurement
# ---------------------------------------------------------------------------

LLM_SIGNAL_SYSTEM = """You are a code quality judge evaluating an LLM's response to a coding task.

Score each facet 0.0-1.0:

CORRECTNESS: Is the generated code syntactically valid and logically sound? Are types correct? Would it compile/parse? Are there obvious bugs? Score low if: syntax errors, type mismatches, undefined variables, logic errors.

CONSISTENCY: Does the code follow consistent patterns? Are naming conventions uniform (all snake_case or all camelCase, not mixed)? Are imports well-organized? Does the style match what a senior developer would expect? Score low if: mixed naming, inconsistent formatting, wildcard imports.

COMPLETENESS: Are edge cases handled? Is there error handling (not bare except)? Are there tests or test suggestions? Is there documentation? Score low if: no error handling, no input validation, missing docstrings, TODO/FIXME markers.

ROBUSTNESS: Does the code handle failures gracefully? Are resources properly managed (context managers, cleanup)? Is there input validation? Score low if: bare except clauses, no resource cleanup, eval/exec usage, hardcoded secrets.

Return ONLY a JSON object:
{"correctness": 0.0, "consistency": 0.0, "completeness": 0.0, "robustness": 0.0}"""

LLM_SIGNAL_USER_TEMPLATE = """Project context (first 2000 chars):
{soul}

Task/prompt:
{prompt}

Response to evaluate:
{response}"""


def parse_llm_signal(llm_response: str) -> Optional[Dict[str, float]]:
    """
    Parse a JSON object from an LLM response.

    Handles both raw JSON and JSON wrapped in markdown code blocks.
    Returns None if the response cannot be parsed.
    """
    if not llm_response:
        return None

    text = llm_response.strip()

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.S)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Last resort: find first { ... } block
        brace_match = re.search(r"\{[^{}]+\}", text)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
            except (json.JSONDecodeError, ValueError):
                return None
        else:
            return None

    # Validate: must have all four facets, all floats in [0, 1]
    facets = ("correctness", "consistency", "completeness", "robustness")
    result: Dict[str, float] = {}
    for f in facets:
        val = data.get(f)
        if val is None:
            return None
        try:
            val = float(val)
        except (TypeError, ValueError):
            return None
        result[f] = max(0.0, min(1.0, val))

    return result


def blend_signals(
    regex_signal: Signal,
    llm_scores: Dict[str, float],
    llm_weight: float = 0.6,
    trace_ids: Optional[List[str]] = None,
) -> Signal:
    """
    Blend regex and LLM measurements.

    Default split: 60% LLM, 40% regex.
    """
    rw = 1.0 - llm_weight

    return Signal(
        correctness=rw * regex_signal.correctness
        + llm_weight * llm_scores["correctness"],
        consistency=rw * regex_signal.consistency
        + llm_weight * llm_scores["consistency"],
        completeness=rw * regex_signal.completeness
        + llm_weight * llm_scores["completeness"],
        robustness=rw * regex_signal.robustness + llm_weight * llm_scores["robustness"],
        trace_ids=trace_ids or [],
    )


# ---------------------------------------------------------------------------
# Unified measurement entry point
# ---------------------------------------------------------------------------


def measure(
    text: str,
    llm_func: Optional[LLMFunc] = None,
    soul_text: str = "",
    prompt: str = "",
    trace_ids: Optional[List[str]] = None,
    llm_weight: float = 0.6,
) -> Signal:
    """
    Measure Code Quality Signal in a text response.

    Always runs regex measurement.  If *llm_func* is provided, also calls
    the LLM for a higher-quality reading and blends the two.  Falls back
    to regex-only silently if the LLM call fails.

    Parameters
    ----------
    text : str
        The response text to evaluate.
    llm_func : callable, optional
        ``llm_func(prompt: str, system: str) -> str``
    soul_text : str
        Project context (SOUL.md) — first 2000 chars sent to the LLM judge.
    prompt : str
        The original prompt the response was generated for.
    trace_ids : list[str], optional
        Trace IDs to attach to the resulting Signal.
    llm_weight : float
        Weight for LLM scores when blending (default 0.6).
    """
    regex_signal = measure_regex(text)

    if llm_func is None:
        if trace_ids:
            regex_signal.trace_ids = trace_ids
        return regex_signal

    # Attempt LLM measurement
    try:
        user_prompt = LLM_SIGNAL_USER_TEMPLATE.format(
            soul=soul_text[:2000],
            prompt=prompt,
            response=text,
        )
        raw = llm_func(user_prompt, LLM_SIGNAL_SYSTEM)
        llm_scores = parse_llm_signal(raw)

        if llm_scores is not None:
            return blend_signals(
                regex_signal, llm_scores, llm_weight=llm_weight, trace_ids=trace_ids
            )
    except Exception as exc:
        log.debug("LLM signal measurement failed, using regex-only: %s", exc)

    if trace_ids:
        regex_signal.trace_ids = trace_ids
    return regex_signal


# ---------------------------------------------------------------------------
# Signal tracker — rolling window analytics
# ---------------------------------------------------------------------------


class SignalTracker:
    """
    Maintains a rolling window of Signal readings and provides trend
    analytics.

    Unchanged from v1 — the tracker is infrastructure, only the Signal
    contents changed (CQS facets instead of consciousness facets).
    """

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self.signals: List[Signal] = []

    def record(self, signal: Signal) -> None:
        """Add a signal to the rolling window."""
        self.signals.append(signal)
        if len(self.signals) > self.window_size:
            self.signals = self.signals[-self.window_size :]

    def recent_health(self) -> float:
        """Average health of the last 5 signals (or all if < 5)."""
        recent = self.signals[-5:]
        if not recent:
            return 0.5
        return sum(s.health for s in recent) / len(recent)

    def trend(self) -> str:
        """
        Compare the last 5 signals against the previous 5.

        Returns "improving", "stable", or "declining".
        """
        if len(self.signals) < 2:
            return "stable"

        recent = self.signals[-5:]
        previous = (
            self.signals[-10:-5] if len(self.signals) >= 10 else self.signals[:-5]
        )

        if not previous:
            return "stable"

        recent_avg = sum(s.health for s in recent) / len(recent)
        prev_avg = sum(s.health for s in previous) / len(previous)

        diff = recent_avg - prev_avg
        if diff > 0.05:
            return "improving"
        if diff < -0.05:
            return "declining"
        return "stable"

    def recovery_rate(self) -> float:
        """
        How quickly health restores after dips.

        Looks at transitions from below-0.5 to above-0.5.  Returns average
        improvement per step during recovery episodes, or 0.0 if no
        recoveries have been observed.
        """
        if len(self.signals) < 3:
            return 0.0

        recovery_deltas: List[float] = []
        in_dip = False

        for i in range(1, len(self.signals)):
            prev_h = self.signals[i - 1].health
            curr_h = self.signals[i].health

            if prev_h < 0.5:
                in_dip = True

            if in_dip and curr_h > prev_h:
                recovery_deltas.append(curr_h - prev_h)

            if curr_h >= 0.5:
                in_dip = False

        if not recovery_deltas:
            return 0.0
        return sum(recovery_deltas) / len(recovery_deltas)

    def to_dict(self) -> Dict:
        return {
            "window_size": self.window_size,
            "count": len(self.signals),
            "recent_health": round(self.recent_health(), 4),
            "trend": self.trend(),
            "recovery_rate": round(self.recovery_rate(), 4),
            "signals": [s.to_dict() for s in self.signals[-10:]],  # last 10 only
        }
