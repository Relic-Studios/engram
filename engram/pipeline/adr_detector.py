"""
engram.pipeline.adr_detector — Detect architectural decisions in LLM responses.

Scans response text for patterns that indicate a significant design
choice was made (e.g., "decided to use X over Y", "switched from A
to B", "chose X because...").  When detected, returns structured
ADR candidates that the after-pipeline can auto-log as high-salience
architecture_decision traces.

Detection is purely heuristic (regex + keyword patterns) — no LLM
call required.  This keeps it fast enough to run on every exchange.

Reference: Michael Nygard ADR format (Context, Options, Decision,
Consequences).  We extract what we can and leave the rest for the
agent to fill in later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# ADR candidate dataclass
# ---------------------------------------------------------------------------


@dataclass
class ADRCandidate:
    """A detected architectural decision candidate.

    Fields map loosely to the Nygard ADR format.  The detector fills
    in what it can extract; the rest defaults to empty.
    """

    decision: str  # What was decided (the core statement)
    context: str = ""  # Why this decision was made
    options: str = ""  # Alternatives that were considered
    consequences: str = ""  # Trade-offs mentioned
    confidence: float = 0.5  # How confident we are this is a real ADR (0-1)
    source_line: str = ""  # The original line that triggered detection


# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

# Patterns that indicate a design choice.  Each tuple is
# (compiled_regex, confidence_boost, description).
# Confidence starts at 0.0 and accumulates from matching patterns.

_DECISION_PATTERNS: List[tuple] = [
    # Explicit decision language
    (
        re.compile(
            r"\b(?:decided|choosing|chose|opted|selected|picked|went with|going with|"
            r"switching to|migrated to|moved to|adopted|settled on)\b"
            r".*?\b(?:instead of|over|rather than|because|since|due to|for)\b",
            re.IGNORECASE,
        ),
        0.7,
        "explicit_decision_with_reason",
    ),
    # "Use X instead of Y" pattern
    (
        re.compile(
            r"\buse\s+\w+(?:\s+\w+){0,3}\s+(?:instead of|over|rather than)\s+\w+",
            re.IGNORECASE,
        ),
        0.6,
        "use_x_over_y",
    ),
    # "Decided to ..." (strong signal)
    (
        re.compile(
            r"\b(?:decided to|decision to|design decision|architectural decision)\b",
            re.IGNORECASE,
        ),
        0.6,
        "decided_to",
    ),
    # "Switched from X to Y"
    (
        re.compile(
            r"\b(?:switched|migrated|moved|transitioned|converted)\s+from\s+\w+\s+to\s+\w+",
            re.IGNORECASE,
        ),
        0.5,
        "switched_from_to",
    ),
    # "Chose X because ..."
    (
        re.compile(
            r"\b(?:chose|selected|picked|opted for)\b.*\bbecause\b",
            re.IGNORECASE,
        ),
        0.6,
        "chose_because",
    ),
    # Trade-off language (boosts confidence when combined with above)
    (
        re.compile(
            r"\b(?:trade-?off|downside|drawback|limitation|sacrifice|compromise|"
            r"at the cost of|the risk is|caveat)\b",
            re.IGNORECASE,
        ),
        0.2,
        "tradeoff_language",
    ),
    # Architecture keywords (boosts when combined)
    (
        re.compile(
            r"\b(?:architecture|architectural|infrastructure|schema|database|"
            r"api design|endpoint design|data model|service layer|"
            r"microservice|monolith|event-driven|message queue|"
            r"authentication|authorization|caching strategy|"
            r"deployment|ci/cd|testing strategy)\b",
            re.IGNORECASE,
        ),
        0.15,
        "architecture_keyword",
    ),
    # Comparison / alternatives considered
    (
        re.compile(
            r"\b(?:considered|evaluated|compared|weighed|alternatives?|options? (?:were|are|include))\b",
            re.IGNORECASE,
        ),
        0.25,
        "alternatives_considered",
    ),
    # "We should / we will / I recommend" (weaker signal)
    (
        re.compile(
            r"\b(?:we should|we will|i recommend|recommend using|"
            r"the approach is|the strategy is|the plan is)\b",
            re.IGNORECASE,
        ),
        0.15,
        "recommendation",
    ),
]

# Minimum confidence threshold for a candidate to be returned
_MIN_CONFIDENCE = 0.5

# Patterns that indicate this is NOT an ADR (false positive filters)
_ANTI_PATTERNS = [
    re.compile(r"^\s*(?:#|//|/\*|\*|<!--)", re.MULTILINE),  # Code comments
    re.compile(r"```", re.DOTALL),  # Inside code blocks
]


# ---------------------------------------------------------------------------
# Sentence extraction helper
# ---------------------------------------------------------------------------


def _extract_sentences(text: str) -> List[str]:
    """Split text into sentences, filtering out code blocks."""
    # Remove code blocks first
    cleaned = re.sub(r"```[\s\S]*?```", " ", text)
    # Remove inline code
    cleaned = re.sub(r"`[^`]+`", " ", cleaned)
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+|\n\n+|\n(?=[A-Z])", cleaned)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------


def detect_adr_candidates(
    response: str,
    min_confidence: float = 0.0,
) -> List[ADRCandidate]:
    """Detect architectural decision candidates in an LLM response.

    Scans the response for patterns indicating design choices,
    technology selections, or architectural trade-offs.  Returns
    candidates sorted by confidence descending.

    Parameters
    ----------
    response:
        The full LLM response text.
    min_confidence:
        Minimum confidence threshold (0-1).  Uses module default
        if 0.

    Returns
    -------
    list[ADRCandidate]
        Detected ADR candidates, sorted by confidence descending.
        Empty list if no architectural decisions detected.
    """
    threshold = min_confidence or _MIN_CONFIDENCE
    sentences = _extract_sentences(response)
    candidates: List[ADRCandidate] = []
    seen_decisions: set = set()

    for sentence in sentences:
        confidence = 0.0
        matched_patterns: List[str] = []

        for pattern, boost, name in _DECISION_PATTERNS:
            if pattern.search(sentence):
                confidence += boost
                matched_patterns.append(name)

        if confidence >= threshold:
            # Extract the decision (the sentence itself, cleaned up)
            decision = sentence.strip()
            if len(decision) > 300:
                decision = decision[:300] + "..."

            # Deduplicate similar decisions
            decision_key = decision[:80].lower()
            if decision_key in seen_decisions:
                continue
            seen_decisions.add(decision_key)

            # Try to extract context and options from surrounding text
            context = _extract_context(response, sentence)
            options = _extract_options(sentence)

            candidate = ADRCandidate(
                decision=decision,
                context=context,
                options=options,
                confidence=min(confidence, 1.0),
                source_line=sentence[:200],
            )

            # Check for consequence/trade-off language nearby
            consequences = _extract_consequences(response, sentence)
            if consequences:
                candidate.consequences = consequences

            candidates.append(candidate)

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def _extract_context(full_text: str, decision_sentence: str) -> str:
    """Try to extract context from sentences preceding the decision."""
    sentences = _extract_sentences(full_text)
    try:
        idx = next(i for i, s in enumerate(sentences) if decision_sentence[:50] in s)
    except StopIteration:
        return ""

    # Look at 1-2 sentences before for context
    context_parts = []
    for i in range(max(0, idx - 2), idx):
        s = sentences[i].strip()
        if len(s) > 20:
            context_parts.append(s)

    return " ".join(context_parts) if context_parts else ""


def _extract_options(sentence: str) -> str:
    """Try to extract alternatives from the decision sentence."""
    # "X instead of Y", "X over Y", "X rather than Y"
    m = re.search(
        r"(\w[\w\s]*?)\s+(?:instead of|over|rather than)\s+(\w[\w\s]*?)(?:[.,;]|$)",
        sentence,
        re.IGNORECASE,
    )
    if m:
        chosen = m.group(1).strip()
        rejected = m.group(2).strip()
        return f"1. {chosen} (chosen) 2. {rejected} (rejected)"

    # "switched from X to Y"
    m = re.search(
        r"(?:switched|migrated|moved)\s+from\s+(\w[\w\s]*?)\s+to\s+(\w[\w\s]*?)(?:[.,;]|$)",
        sentence,
        re.IGNORECASE,
    )
    if m:
        old = m.group(1).strip()
        new = m.group(2).strip()
        return f"1. {new} (chosen) 2. {old} (previous)"

    return ""


def _extract_consequences(full_text: str, decision_sentence: str) -> str:
    """Try to extract consequences from sentences following the decision."""
    sentences = _extract_sentences(full_text)
    try:
        idx = next(i for i, s in enumerate(sentences) if decision_sentence[:50] in s)
    except StopIteration:
        return ""

    # Look at 1-2 sentences after for consequences
    consequence_patterns = re.compile(
        r"\b(?:trade-?off|downside|drawback|however|but|caveat|"
        r"the risk|consequence|this means|as a result|"
        r"the cost|limitation|sacrifice)\b",
        re.IGNORECASE,
    )

    for i in range(idx + 1, min(idx + 3, len(sentences))):
        s = sentences[i].strip()
        if consequence_patterns.search(s):
            return s

    return ""
