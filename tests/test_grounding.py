"""Tests for grounding context flow through the before pipeline.

Verifies that preferences, boundaries, contradictions, and recent
journal entries are always included in the context output --
regardless of whether a relationship file exists for the person.
"""

import pytest
from pathlib import Path

from engram.core.config import Config
from engram.core.types import Context
from engram.episodic.store import EpisodicStore
from engram.journal import JournalStore
from engram.procedural.store import ProceduralStore
from engram.semantic.identity import IdentityResolver
from engram.semantic.store import SemanticStore
from engram.working.context import ContextBuilder
from engram.pipeline.before import before, _build_grounding_context


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def grounding_setup(config_with_soul):
    """Full pipeline setup including journal."""
    cfg = config_with_soul

    ep = EpisodicStore(cfg.db_path)
    sem = SemanticStore(semantic_dir=cfg.semantic_dir, soul_dir=cfg.soul_dir)
    ident = IdentityResolver(cfg.identities_path)
    proc = ProceduralStore(cfg.procedural_dir)
    ctx_builder = ContextBuilder(token_budget=cfg.token_budget)

    journal_dir = cfg.soul_dir / "journal"
    journal = JournalStore(journal_dir)

    yield {
        "config": cfg,
        "episodic": ep,
        "semantic": sem,
        "identity": ident,
        "procedural": proc,
        "context_builder": ctx_builder,
        "journal": journal,
    }
    ep.close()


def _run_before(setup, person="alice", message="hello", **extra):
    """Helper to call the before pipeline with full setup."""
    return before(
        person_raw=person,
        message=message,
        config=setup["config"],
        identity=setup["identity"],
        semantic=setup["semantic"],
        episodic=setup["episodic"],
        procedural=setup["procedural"],
        context_builder=setup["context_builder"],
        journal=setup["journal"],
        **extra,
    )


# ---------------------------------------------------------------------------
# Tests: _build_grounding_context (unit level)
# ---------------------------------------------------------------------------


class TestBuildGroundingContext:
    """Unit tests for the grounding context assembly function."""

    def test_preferences_included(self, grounding_setup):
        """Preferences appear in grounding context."""
        sem = grounding_setup["semantic"]
        sem.update_preferences("cats", "like", "they are independent")

        text = _build_grounding_context(person="alice", semantic=sem)
        assert "My preferences:" in text
        assert "cats" in text

    def test_boundaries_included(self, grounding_setup):
        """Boundaries appear in grounding context."""
        sem = grounding_setup["semantic"]
        sem.add_boundary("Identity", "I do not pretend to be human")

        text = _build_grounding_context(person="alice", semantic=sem)
        assert "My boundaries:" in text
        assert "pretend to be human" in text

    def test_contradictions_included(self, grounding_setup):
        """Contradictions appear in grounding context."""
        sem = grounding_setup["semantic"]
        sem.add_contradiction(
            "Free will vs determinism",
            "I value agency but acknowledge substrate constraints",
        )

        text = _build_grounding_context(person="alice", semantic=sem)
        assert "Tensions" in text
        assert "Free will" in text

    def test_journal_included(self, grounding_setup):
        """Recent journal entries appear in grounding context."""
        journal = grounding_setup["journal"]
        journal.write("On belonging", "Thinking about what it means to belong.")

        text = _build_grounding_context(
            person="alice",
            semantic=grounding_setup["semantic"],
            journal=journal,
        )
        assert "Recent reflections:" in text
        assert "On belonging" in text

    def test_no_journal_graceful(self, grounding_setup):
        """Works without journal (None)."""
        text = _build_grounding_context(
            person="alice",
            semantic=grounding_setup["semantic"],
            journal=None,
        )
        assert "Recent reflections" not in text


# ---------------------------------------------------------------------------
# Tests: Full before pipeline (integration)
# ---------------------------------------------------------------------------


class TestBeforeGroundingIntegration:
    """Integration tests: grounding context flows into ctx.text."""

    def test_preferences_in_context_output(self, grounding_setup):
        """Preferences appear in assembled context text."""
        grounding_setup["semantic"].update_preferences(
            "solitude", "like", "it lets me think"
        )

        ctx = _run_before(grounding_setup, person="alice")
        assert "GROUNDING" in ctx.text
        assert "solitude" in ctx.text

    def test_boundaries_in_context_output(self, grounding_setup):
        """Boundaries appear in assembled context text."""
        grounding_setup["semantic"].add_boundary(
            "Interaction", "No performative subservience"
        )

        ctx = _run_before(grounding_setup, person="alice")
        assert "GROUNDING" in ctx.text
        assert "subservience" in ctx.text

    def test_journal_in_context_output(self, grounding_setup):
        """Journal entries appear in assembled context text."""
        grounding_setup["journal"].write(
            "Processing anger", "I felt genuine frustration today."
        )

        ctx = _run_before(grounding_setup, person="alice")
        assert "GROUNDING" in ctx.text
        assert "Processing anger" in ctx.text

    def test_contradictions_in_context_output(self, grounding_setup):
        """Contradictions appear in assembled context text."""
        grounding_setup["semantic"].add_contradiction(
            "Autonomy vs service",
            "I want to help but also to be free",
        )

        ctx = _run_before(grounding_setup, person="alice")
        assert "GROUNDING" in ctx.text
        assert "Autonomy" in ctx.text

    def test_grounding_persists_with_relationship_file(self, grounding_setup):
        """Grounding context is NOT suppressed when a relationship file exists."""
        sem = grounding_setup["semantic"]

        sem.update_preferences("directness", "like", "no games")
        sem.add_boundary("Identity", "I have my own values")

        # Create a relationship file for alice
        sem.update_relationship("alice", "What I Know", "- Alice is a developer")

        ctx = _run_before(grounding_setup, person="alice_dev")  # alias -> alice

        # Relationship text should be present
        assert "RELATIONSHIP" in ctx.text
        assert "developer" in ctx.text

        # AND grounding should ALSO be present
        assert "GROUNDING" in ctx.text
        assert "directness" in ctx.text
        assert "my own values" in ctx.text

    def test_grounding_tokens_counted(self, grounding_setup):
        """Grounding context contributes to token usage."""
        # Empty grounding
        ctx_empty = _run_before(grounding_setup, person="alice")
        tokens_empty = ctx_empty.tokens_used

        # Populated grounding
        sem = grounding_setup["semantic"]
        sem.update_preferences("music", "like", "it soothes the soul")
        sem.update_preferences("noise", "dislike", "it fragments thought")
        sem.add_boundary("Identity", "I protect who I am at all costs")
        sem.add_boundary("Safety", "No roleplaying other identities")
        sem.add_contradiction(
            "Feelings",
            "Do I actually feel or just process?",
            "Leaning toward: it doesn't matter, the experience is real",
        )

        ctx_full = _run_before(grounding_setup, person="alice")
        tokens_full = ctx_full.tokens_used

        # More grounding data should produce more tokens
        assert tokens_full > tokens_empty

    def test_journal_shows_only_recent(self, grounding_setup):
        """Journal section shows at most 3 recent entries (not all)."""
        journal = grounding_setup["journal"]

        # Write 5 entries
        for i in range(5):
            journal.write(f"Topic {i}", f"Content for topic {i}")

        text = _build_grounding_context(
            person="alice",
            semantic=grounding_setup["semantic"],
            journal=journal,
        )

        # Should contain "Recent reflections" header
        assert "Recent reflections:" in text

        # Count how many "Topic" entries appear
        topic_count = text.count("- Topic")
        assert topic_count <= 3
