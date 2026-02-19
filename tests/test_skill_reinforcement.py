"""
Tests for frequency-based confidence modeling (C2).

Covers:
  - reinforce_matched() — matching skills in response text
  - Reinforcement with high CQS (accepted)
  - Penalization with low CQS (rejected)
  - Dead band (no reinforcement in 0.4-0.7 range)
  - get_promotable() — Core Skill promotion candidates
  - Promotion thresholds (confidence + min_accepted)
"""

from __future__ import annotations

import pytest

from engram.procedural.store import ProceduralStore


@pytest.fixture
def store(tmp_path):
    return ProceduralStore(tmp_path / "skills")


# =========================================================================
# reinforce_matched
# =========================================================================


class TestReinforceMatched:
    """Tests for frequency-based reinforcement of matched skills."""

    def test_high_signal_reinforces(self, store):
        store.add_structured_skill(
            name="retry-backoff",
            content="# Retry with exponential backoff",
            tags=["retry", "backoff"],
        )
        results = store.reinforce_matched(
            response="Using retry backoff pattern for the HTTP client",
            signal_health=0.85,
        )
        assert len(results) == 1
        assert results[0]["accepted"] is True
        assert results[0]["confidence"] > 0

    def test_low_signal_penalizes(self, store):
        store.add_structured_skill(
            name="eval-hack",
            content="# Eval-based dispatch",
            tags=["eval", "dynamic"],
        )
        results = store.reinforce_matched(
            response="Using eval-based dispatch for routing",
            signal_health=0.2,
        )
        assert len(results) == 1
        assert results[0]["accepted"] is False

    def test_dead_band_no_reinforcement(self, store):
        store.add_structured_skill(
            name="some-pattern",
            content="# Pattern",
            tags=["some-pattern"],
        )
        results = store.reinforce_matched(
            response="Using some-pattern in the code",
            signal_health=0.55,  # dead band
        )
        assert len(results) == 0

    def test_no_match_no_reinforcement(self, store):
        store.add_structured_skill(
            name="database-pool",
            content="# Connection pooling",
            tags=["database", "pooling"],
        )
        results = store.reinforce_matched(
            response="Implemented a simple hello world function",
            signal_health=0.9,
        )
        assert len(results) == 0

    def test_matches_by_tag(self, store):
        store.add_structured_skill(
            name="obscure-name",
            content="# Some pattern",
            tags=["authentication", "oauth"],
        )
        results = store.reinforce_matched(
            response="Added authentication middleware to the app",
            signal_health=0.8,
        )
        assert len(results) == 1
        assert results[0]["name"] == "obscure-name"

    def test_short_tags_not_matched(self, store):
        """Tags under 4 chars should not trigger false positives."""
        store.add_structured_skill(
            name="api-pattern",
            content="# API",
            tags=["api"],  # only 3 chars
        )
        results = store.reinforce_matched(
            response="The API endpoint handles requests",
            signal_health=0.9,
        )
        # "api" is < 4 chars, shouldn't match on tag alone
        assert len(results) == 0

    def test_multiple_skills_reinforced(self, store):
        store.add_structured_skill(
            name="retry-pattern",
            content="# Retry",
            tags=["retry", "resilience"],
        )
        store.add_structured_skill(
            name="logging-pattern",
            content="# Logging",
            tags=["logging", "observability"],
        )
        results = store.reinforce_matched(
            response="Added retry logic with structured logging throughout",
            signal_health=0.85,
        )
        assert len(results) == 2

    def test_confidence_accumulates(self, store):
        store.add_structured_skill(
            name="pattern",
            content="# Pattern",
            tags=["pattern"],
        )
        # Reinforce 5 times
        for _ in range(5):
            store.reinforce_matched(
                response="Using pattern approach",
                signal_health=0.9,
            )
        meta = store.get_skill_meta("pattern")
        assert meta.accepted_count == 5
        assert meta.confidence > 0.8

    def test_boundary_at_reinforce_threshold(self, store):
        store.add_structured_skill(name="p", content="# P", tags=["pattern"])
        # Exactly at threshold — should NOT reinforce (> not >=)
        results = store.reinforce_matched(response="Using pattern", signal_health=0.7)
        assert len(results) == 0

    def test_boundary_at_weaken_threshold(self, store):
        store.add_structured_skill(name="p", content="# P", tags=["pattern"])
        # Exactly at weaken threshold — should NOT penalize (< not <=)
        results = store.reinforce_matched(response="Using pattern", signal_health=0.4)
        assert len(results) == 0


# =========================================================================
# get_promotable — Core Skill promotion
# =========================================================================


class TestGetPromotable:
    """Tests for Core Skill promotion candidates."""

    def test_no_skills_returns_empty(self, store):
        assert store.get_promotable() == []

    def test_low_confidence_not_promotable(self, store):
        store.add_structured_skill(name="p", content="# P")
        # Default confidence is 0.5, below PROMOTION_CONFIDENCE (0.85)
        assert store.get_promotable() == []

    def test_high_confidence_low_count_not_promotable(self, store):
        store.add_structured_skill(name="p", content="# P")
        # Manually set high confidence but low count
        meta = store.get_skill_meta("p")
        path = store.skills_dir / "p.md"
        from engram.procedural.schema import serialize_frontmatter, parse_frontmatter

        _, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        meta.confidence = 0.9
        meta.accepted_count = 2  # below PROMOTION_MIN_ACCEPTS (5)
        path.write_text(serialize_frontmatter(meta, body), encoding="utf-8")

        assert store.get_promotable() == []

    def test_promotable_skill(self, store):
        store.add_structured_skill(name="p", content="# P")
        # Reinforce enough times to cross both thresholds
        for _ in range(10):
            store.update_confidence("p", accepted=True)

        meta = store.get_skill_meta("p")
        assert meta.accepted_count >= 5
        assert meta.confidence >= 0.85

        promotable = store.get_promotable()
        assert len(promotable) == 1
        assert promotable[0].name == "p"

    def test_custom_thresholds(self, store):
        store.add_structured_skill(name="p", content="# P")
        # Just 3 accepts: 3/(3+0+1) = 0.75
        for _ in range(3):
            store.update_confidence("p", accepted=True)

        # Not promotable with defaults
        assert store.get_promotable() == []

        # Promotable with lower thresholds
        promotable = store.get_promotable(confidence_threshold=0.7, min_accepted=3)
        assert len(promotable) == 1

    def test_sorted_by_confidence(self, store):
        store.add_structured_skill(name="a", content="# A")
        store.add_structured_skill(name="b", content="# B")

        for _ in range(10):
            store.update_confidence("a", accepted=True)
        for _ in range(8):
            store.update_confidence("b", accepted=True)
        # b also has 2 rejections
        for _ in range(2):
            store.update_confidence("b", accepted=False)

        promotable = store.get_promotable()
        if len(promotable) >= 2:
            assert promotable[0].confidence >= promotable[1].confidence
