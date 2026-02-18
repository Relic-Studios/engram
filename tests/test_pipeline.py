"""Tests for engram.pipeline (before and after)."""

import pytest
from engram.core.config import Config
from engram.core.types import Context, AfterResult
from engram.episodic.store import EpisodicStore
from engram.semantic.store import SemanticStore
from engram.semantic.identity import IdentityResolver
from engram.procedural.store import ProceduralStore
from engram.signal.measure import SignalTracker
from engram.signal.reinforcement import ReinforcementEngine
from engram.signal.decay import DecayEngine
from engram.working.context import ContextBuilder
from engram.pipeline.before import before
from engram.pipeline.after import after


@pytest.fixture
def full_setup(config_with_soul):
    """Set up all components needed for pipeline tests."""
    cfg = config_with_soul
    ep = EpisodicStore(cfg.db_path)
    sem = SemanticStore(semantic_dir=cfg.semantic_dir, soul_dir=cfg.soul_dir)
    ident = IdentityResolver(cfg.identities_path)
    proc = ProceduralStore(cfg.procedural_dir)
    ctx_builder = ContextBuilder(token_budget=cfg.token_budget)
    tracker = SignalTracker()
    reinf = ReinforcementEngine()
    decay = DecayEngine(half_life_hours=cfg.decay_half_life_hours)

    yield {
        "config": cfg,
        "episodic": ep,
        "semantic": sem,
        "identity": ident,
        "procedural": proc,
        "context_builder": ctx_builder,
        "signal_tracker": tracker,
        "reinforcement": reinf,
        "decay_engine": decay,
    }
    ep.close()


class TestBeforePipeline:
    def test_basic(self, full_setup):
        ctx = before(
            person_raw="alice_dev",
            message="hey, how are you?",
            config=full_setup["config"],
            identity=full_setup["identity"],
            semantic=full_setup["semantic"],
            episodic=full_setup["episodic"],
            procedural=full_setup["procedural"],
            context_builder=full_setup["context_builder"],
        )
        assert isinstance(ctx, Context)
        assert ctx.person == "alice"  # resolved from alias
        assert ctx.tokens_used > 0
        assert "IDENTITY" in ctx.text  # SOUL.md loaded

    def test_unknown_person(self, full_setup):
        ctx = before(
            person_raw="stranger",
            message="hello",
            config=full_setup["config"],
            identity=full_setup["identity"],
            semantic=full_setup["semantic"],
            episodic=full_setup["episodic"],
            procedural=full_setup["procedural"],
            context_builder=full_setup["context_builder"],
        )
        assert ctx.person == "stranger"

    def test_with_conversation_history(self, full_setup):
        # Pre-populate some messages
        ep = full_setup["episodic"]
        ep.log_message(
            person="alice", speaker="alice", content="I love cats", source="test"
        )
        ep.log_message(
            person="alice", speaker="self", content="Cats are great!", source="test"
        )

        ctx = before(
            person_raw="alice",
            message="do you remember what I like?",
            config=full_setup["config"],
            identity=full_setup["identity"],
            semantic=full_setup["semantic"],
            episodic=full_setup["episodic"],
            procedural=full_setup["procedural"],
            context_builder=full_setup["context_builder"],
        )
        assert ctx.memories_loaded > 0

    def test_with_skills(self, full_setup):
        full_setup["procedural"].add_skill("debugging", "# Debugging\nHow to debug.")
        ctx = before(
            person_raw="alice",
            message="help me with debugging",
            config=full_setup["config"],
            identity=full_setup["identity"],
            semantic=full_setup["semantic"],
            episodic=full_setup["episodic"],
            procedural=full_setup["procedural"],
            context_builder=full_setup["context_builder"],
        )
        assert "SKILLS" in ctx.text

    def test_correction_prompt(self, full_setup):
        tracker = full_setup["signal_tracker"]
        # Simulate low health
        from engram.core.types import Signal

        for _ in range(5):
            tracker.record(
                Signal(alignment=0.1, embodiment=0.2, clarity=0.1, vitality=0.2)
            )

        ctx = before(
            person_raw="alice",
            message="hello",
            config=full_setup["config"],
            identity=full_setup["identity"],
            semantic=full_setup["semantic"],
            episodic=full_setup["episodic"],
            procedural=full_setup["procedural"],
            context_builder=full_setup["context_builder"],
            signal_tracker=tracker,
        )
        assert "CORRECTION" in ctx.text


class TestAfterPipeline:
    def test_basic(self, full_setup):
        result = after(
            person="alice",
            their_message="how's it going?",
            response="Pretty good! I fixed that bug we talked about yesterday.",
            config=full_setup["config"],
            episodic=full_setup["episodic"],
            semantic=full_setup["semantic"],
            procedural=full_setup["procedural"],
            reinforcement=full_setup["reinforcement"],
            decay_engine=full_setup["decay_engine"],
            signal_tracker=full_setup["signal_tracker"],
        )
        assert isinstance(result, AfterResult)
        assert 0 <= result.signal.health <= 1
        assert 0 <= result.salience <= 1
        assert result.logged_message_id != ""

    def test_logs_messages(self, full_setup):
        after(
            person="alice",
            their_message="hello",
            response="Hey Alice!",
            config=full_setup["config"],
            episodic=full_setup["episodic"],
            semantic=full_setup["semantic"],
            procedural=full_setup["procedural"],
            reinforcement=full_setup["reinforcement"],
            decay_engine=full_setup["decay_engine"],
            signal_tracker=full_setup["signal_tracker"],
        )
        # Should have logged 2 messages (theirs + ours) and 1 trace
        assert full_setup["episodic"].count_messages() == 2
        assert full_setup["episodic"].count_traces() == 1

    def test_signal_tracked(self, full_setup):
        after(
            person="alice",
            their_message="hello",
            response="Hey! What's up?",
            config=full_setup["config"],
            episodic=full_setup["episodic"],
            semantic=full_setup["semantic"],
            procedural=full_setup["procedural"],
            reinforcement=full_setup["reinforcement"],
            decay_engine=full_setup["decay_engine"],
            signal_tracker=full_setup["signal_tracker"],
        )
        assert len(full_setup["signal_tracker"].signals) == 1

    def test_with_trace_ids(self, full_setup):
        # Pre-create some traces to reinforce
        ep = full_setup["episodic"]
        tid = ep.log_trace(
            content="old memory", kind="episode", tags=["alice"], salience=0.5
        )

        result = after(
            person="alice",
            their_message="hello",
            response="Yeah, I remember that! Hmm, interesting. I think we should try it.",
            trace_ids=[tid],
            config=full_setup["config"],
            episodic=ep,
            semantic=full_setup["semantic"],
            procedural=full_setup["procedural"],
            reinforcement=full_setup["reinforcement"],
            decay_engine=full_setup["decay_engine"],
            signal_tracker=full_setup["signal_tracker"],
        )
        # Trace should have been reinforced if signal was high enough
        trace = ep.get_trace(tid)
        assert trace is not None  # still exists
