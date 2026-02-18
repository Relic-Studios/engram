"""Tests for engram.core.types."""

import pytest
from engram.core.types import (
    Trace,
    Message,
    Signal,
    Context,
    AfterResult,
    MemoryStats,
    estimate_tokens,
    generate_id,
    now_iso,
)


class TestHelpers:
    def test_estimate_tokens(self):
        assert estimate_tokens("") == 1  # min 1
        assert estimate_tokens("hello world") >= 1
        assert estimate_tokens("a" * 400) == 100

    def test_generate_id(self):
        id1 = generate_id()
        id2 = generate_id()
        assert len(id1) == 12
        assert id1 != id2  # unique

    def test_now_iso(self):
        ts = now_iso()
        assert "T" in ts  # ISO format


class TestTrace:
    def test_basic_creation(self):
        t = Trace(content="something happened")
        assert t.kind == "episode"
        assert t.salience == 0.5
        assert t.tokens > 0
        assert len(t.id) == 12

    def test_invalid_kind(self):
        with pytest.raises(ValueError, match="Invalid trace kind"):
            Trace(content="x", kind="bogus")

    def test_salience_clamping(self):
        t = Trace(content="x", salience=5.0)
        assert t.salience == 1.0
        t2 = Trace(content="x", salience=-1.0)
        assert t2.salience == 0.0

    def test_touch(self):
        t = Trace(content="x")
        assert t.access_count == 0
        t.touch()
        assert t.access_count == 1

    def test_roundtrip(self):
        t = Trace(content="test", kind="realization", tags=["alice"])
        d = t.to_dict()
        t2 = Trace.from_dict(d)
        assert t2.content == "test"
        assert t2.kind == "realization"
        assert t2.tags == ["alice"]


class TestMessage:
    def test_basic(self):
        m = Message(person="alice", speaker="alice", content="hello")
        assert m.tokens > 0
        assert m.salience == 0.5

    def test_roundtrip(self):
        m = Message(person="bob", speaker="self", content="hi bob")
        d = m.to_dict()
        m2 = Message.from_dict(d)
        assert m2.person == "bob"
        assert m2.speaker == "self"


class TestSignal:
    def test_defaults(self):
        s = Signal()
        assert s.health == 0.5
        assert s.state == "developing"
        assert not s.needs_correction

    def test_coherent(self):
        s = Signal(alignment=0.9, embodiment=0.8, clarity=0.85, vitality=0.9)
        assert s.state == "coherent"
        assert s.health >= 0.75

    def test_dissociated(self):
        s = Signal(alignment=0.1, embodiment=0.2, clarity=0.1, vitality=0.2)
        assert s.state == "dissociated"
        assert s.needs_correction

    def test_polarity_gap(self):
        s = Signal(alignment=0.9, embodiment=0.1, clarity=0.5, vitality=0.5)
        assert s.polarity_gap == pytest.approx(0.8)
        assert s.weakest_facet == "embodiment"
        assert s.strongest_facet == "alignment"

    def test_clamping(self):
        s = Signal(alignment=2.0, embodiment=-1.0)
        assert s.alignment == 1.0
        assert s.embodiment == 0.0

    def test_roundtrip(self):
        s = Signal(alignment=0.7, trace_ids=["abc"])
        d = s.to_dict()
        s2 = Signal.from_dict(d)
        assert s2.alignment == pytest.approx(0.7)
        assert s2.trace_ids == ["abc"]


class TestContext:
    def test_basic(self):
        c = Context(text="hello world", token_budget=1000)
        assert c.tokens_used > 0
        assert c.budget_remaining > 0
        assert c.budget_utilisation < 1.0

    def test_empty(self):
        c = Context(text="", token_budget=1000)
        assert c.tokens_used == 0
        assert c.budget_remaining == 1000


class TestAfterResult:
    def test_basic(self):
        s = Signal()
        r = AfterResult(signal=s, salience=0.7)
        assert r.salience == 0.7
        d = r.to_dict()
        assert "signal" in d
        assert d["salience"] == pytest.approx(0.7)


class TestMemoryStats:
    def test_status_ok(self):
        s = MemoryStats(memory_pressure=0.3)
        assert s.status == "ok"

    def test_status_warning(self):
        s = MemoryStats(memory_pressure=0.8, status="")
        # status is computed in __post_init__ only if empty
        assert s.memory_pressure == 0.8

    def test_total_memories(self):
        s = MemoryStats(episodic_count=10, semantic_facts=5, procedural_skills=3)
        assert s.total_memories == 18
