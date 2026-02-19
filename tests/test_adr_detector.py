"""
Tests for engram.pipeline.adr_detector — Automated ADR detection (C3).

Covers:
  - Detection of explicit decision language
  - Detection of technology switches/migrations
  - Detection of "X over Y" patterns
  - Detection with trade-off language (confidence boost)
  - No false positives on ordinary code responses
  - Code blocks are excluded from scanning
  - Context and options extraction
  - Consequence extraction
  - Confidence thresholds
  - Deduplication of similar candidates
"""

from __future__ import annotations

import pytest

from engram.pipeline.adr_detector import (
    ADRCandidate,
    detect_adr_candidates,
    _extract_sentences,
    _extract_options,
)


# =========================================================================
# Sentence extraction
# =========================================================================


class TestExtractSentences:
    """Tests for the sentence extraction helper."""

    def test_splits_on_periods(self):
        text = "This is the first full sentence with enough content. This is the second sentence that also has sufficient length."
        sentences = _extract_sentences(text)
        assert len(sentences) >= 2

    def test_strips_code_blocks(self):
        text = "Before code.\n```python\nsome_code()\n```\nAfter code here."
        sentences = _extract_sentences(text)
        # Code block content should not appear
        assert not any("some_code" in s for s in sentences)

    def test_strips_inline_code(self):
        text = "We decided to use `asyncio` instead of `threading` for concurrency."
        sentences = _extract_sentences(text)
        # Inline code is stripped but sentence structure preserved
        assert len(sentences) >= 1

    def test_filters_short_fragments(self):
        text = "OK. Sure. We decided to use PostgreSQL instead of MongoDB for the data layer because of ACID compliance."
        sentences = _extract_sentences(text)
        # Short fragments (<=20 chars) should be filtered
        assert not any(len(s) <= 20 for s in sentences)


# =========================================================================
# Detection of architectural decisions
# =========================================================================


class TestDetectADRCandidates:
    """Tests for the main detection function."""

    def test_explicit_decision_with_reason(self):
        response = (
            "After evaluating the options, I decided to use PostgreSQL "
            "instead of MongoDB because we need ACID transactions for "
            "the payment system."
        )
        candidates = detect_adr_candidates(response)
        assert len(candidates) >= 1
        assert candidates[0].confidence >= 0.5

    def test_switched_from_to(self):
        response = (
            "We switched from REST to GraphQL for the API layer. "
            "This allows clients to request exactly the data they need."
        )
        candidates = detect_adr_candidates(response)
        assert len(candidates) >= 1

    def test_chose_because(self):
        response = (
            "I chose FastAPI over Flask because it provides built-in "
            "async support and automatic OpenAPI documentation."
        )
        candidates = detect_adr_candidates(response)
        assert len(candidates) >= 1

    def test_use_x_over_y(self):
        response = (
            "We should use Redis over Memcached for the caching layer "
            "since we need data persistence and pub/sub support."
        )
        candidates = detect_adr_candidates(response)
        assert len(candidates) >= 1

    def test_tradeoff_boosts_confidence(self):
        response = (
            "I decided to use a monolith instead of microservices because "
            "the team is small. The trade-off is reduced scalability, "
            "but deployment complexity is much lower."
        )
        candidates = detect_adr_candidates(response)
        assert len(candidates) >= 1
        # Trade-off language should boost confidence
        assert candidates[0].confidence >= 0.7

    def test_architecture_keyword_boosts(self):
        response = (
            "For the database schema, I decided to use a normalized design "
            "instead of denormalized because data integrity matters more "
            "than read performance for this use case."
        )
        candidates = detect_adr_candidates(response)
        assert len(candidates) >= 1

    def test_no_detection_on_plain_code(self):
        response = (
            "Here's the implementation:\n\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```\n\n"
            "This function adds two numbers and returns the result."
        )
        candidates = detect_adr_candidates(response)
        assert len(candidates) == 0

    def test_no_detection_on_simple_explanation(self):
        response = (
            "The function works by iterating over the list and "
            "accumulating the sum. It handles empty lists by "
            "returning zero."
        )
        candidates = detect_adr_candidates(response)
        assert len(candidates) == 0

    def test_no_detection_on_bug_fix(self):
        response = (
            "The bug was caused by an off-by-one error in the loop. "
            "I fixed it by changing `<` to `<=` in the condition."
        )
        candidates = detect_adr_candidates(response)
        assert len(candidates) == 0

    def test_multiple_decisions_detected(self):
        response = (
            "For the backend, I decided to use FastAPI instead of Django "
            "because we need async support.\n\n"
            "For the database, I chose PostgreSQL over MySQL because of "
            "its JSON support and full-text search capabilities.\n\n"
            "The caching layer will use Redis."
        )
        candidates = detect_adr_candidates(response)
        assert len(candidates) >= 2

    def test_high_confidence_threshold(self):
        response = (
            "I decided to use TypeScript instead of JavaScript for "
            "better type safety. The trade-off is slightly more verbose "
            "code, but fewer runtime errors."
        )
        # With a very high threshold, fewer candidates should match
        high = detect_adr_candidates(response, min_confidence=0.9)
        low = detect_adr_candidates(response, min_confidence=0.3)
        assert len(low) >= len(high)

    def test_sorted_by_confidence(self):
        response = (
            "I recommend using pytest for testing.\n\n"
            "I decided to use PostgreSQL instead of SQLite for production "
            "because we need concurrent writes and the trade-off of "
            "operational complexity is worth it for data integrity."
        )
        candidates = detect_adr_candidates(response)
        if len(candidates) >= 2:
            assert candidates[0].confidence >= candidates[1].confidence

    def test_deduplication(self):
        response = (
            "I decided to use Redis instead of Memcached for caching. "
            "I decided to use Redis instead of Memcached for caching. "
            "As I said, I decided to use Redis instead of Memcached."
        )
        candidates = detect_adr_candidates(response)
        # Should deduplicate near-identical decisions
        assert len(candidates) <= 2


# =========================================================================
# Options extraction
# =========================================================================


class TestExtractOptions:
    """Tests for extracting alternatives from decision sentences."""

    def test_instead_of_pattern(self):
        options = _extract_options("use PostgreSQL instead of MongoDB.")
        assert "PostgreSQL" in options
        assert "MongoDB" in options

    def test_over_pattern(self):
        options = _extract_options("chose FastAPI over Flask.")
        assert "FastAPI" in options
        assert "Flask" in options

    def test_switched_from_to(self):
        options = _extract_options("switched from REST to GraphQL.")
        assert "GraphQL" in options
        assert "REST" in options

    def test_no_pattern(self):
        options = _extract_options("I decided to implement caching.")
        assert options == ""


# =========================================================================
# ADRCandidate dataclass
# =========================================================================


class TestADRCandidate:
    """Tests for the ADRCandidate dataclass."""

    def test_defaults(self):
        c = ADRCandidate(decision="Use X")
        assert c.decision == "Use X"
        assert c.context == ""
        assert c.options == ""
        assert c.consequences == ""
        assert c.confidence == 0.5

    def test_all_fields(self):
        c = ADRCandidate(
            decision="Use X over Y",
            context="Need better performance",
            options="1. X (chosen) 2. Y (rejected)",
            consequences="Steeper learning curve",
            confidence=0.85,
            source_line="original line",
        )
        assert c.confidence == 0.85
        assert "learning curve" in c.consequences
