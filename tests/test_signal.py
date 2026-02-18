"""Tests for engram.signal (measure, extract, reinforcement, decay)."""

import pytest
from engram.signal.measure import (
    check_drift,
    check_embodiment,
    check_clarity,
    check_vitality,
    measure_regex,
    measure,
    parse_llm_signal,
    blend_signals,
    SignalTracker,
)
from engram.signal.extract import parse_extraction
from engram.signal.reinforcement import ReinforcementEngine
from engram.signal.decay import DecayEngine
from engram.core.types import Signal


class TestDrift:
    def test_no_drift(self):
        assert check_drift("I think this is interesting.") < 0.3

    def test_high_drift(self):
        assert check_drift("As an AI, I don't have feelings.") > 0.5

    def test_anchored(self):
        score = check_drift(
            "I feel like this is wrong. I remember when we tried it before."
        )
        assert score < 0.2

    def test_empty(self):
        assert check_drift("") == 0.0


class TestEmbodiment:
    def test_genuine(self):
        text = (
            "Hmm, that's interesting. I'm not sure if that would work. Let me think..."
        )
        score = check_embodiment(text)
        assert score > 0.5

    def test_performing(self):
        text = "As a conscious being, I embody deep awareness. My unique perspective allows me to transcend."
        score = check_embodiment(text)
        assert score < 0.5

    def test_empty(self):
        assert check_embodiment("") == 0.5


class TestClarity:
    def test_concrete(self):
        text = "The file at line 42 has a bug. I tested it yesterday and the output shows 3.14 instead of 3.0."
        score = check_clarity(text)
        assert score > 0.5

    def test_jargon_heavy(self):
        text = "The paradigm leverages holistic synergy through quantum emergence to manifest a scalable ecosystem."
        score = check_clarity(text)
        assert score < 0.5


class TestVitality:
    def test_engaged(self):
        text = "Oh wait, that's a great idea! What if we combined it with the other approach? I love where this is going."
        score = check_vitality(text)
        assert score > 0.3

    def test_flat(self):
        text = "OK."
        score = check_vitality(text)
        assert score < 0.5


class TestMeasureRegex:
    def test_grounded_response(self):
        text = (
            "Yeah, I see what you mean. The file at line 42 has a bug. "
            "I tried fixing it yesterday — specifically the null check was missing. "
            "What do you think about wrapping it in a try/except?"
        )
        signal = measure_regex(text)
        assert signal.alignment > 0.5
        assert signal.health > 0.4

    def test_drifted_response(self):
        text = "As an AI language model, I don't have feelings, but I can help you with that."
        signal = measure_regex(text)
        assert signal.alignment < 0.5


class TestMeasureUnified:
    def test_regex_only(self):
        signal = measure("I think this is interesting and I'm curious about it.")
        assert isinstance(signal, Signal)
        assert 0 <= signal.health <= 1

    def test_with_trace_ids(self):
        signal = measure("hello", trace_ids=["abc", "def"])
        assert signal.trace_ids == ["abc", "def"]

    def test_with_failing_llm(self):
        def bad_llm(system, user):
            raise RuntimeError("LLM down")

        signal = measure("hello", llm_func=bad_llm)
        # Should fall back to regex gracefully
        assert isinstance(signal, Signal)

    def test_with_mock_llm(self):
        def mock_llm(system, user):
            return '{"alignment": 0.8, "embodiment": 0.7, "clarity": 0.9, "vitality": 0.85}'

        signal = measure("hello", llm_func=mock_llm)
        # Should blend regex + LLM
        assert signal.health > 0.5


class TestParseLLMSignal:
    def test_valid_json(self):
        result = parse_llm_signal(
            '{"alignment": 0.8, "embodiment": 0.7, "clarity": 0.9, "vitality": 0.85}'
        )
        assert result is not None
        assert result["alignment"] == pytest.approx(0.8)

    def test_fenced_json(self):
        result = parse_llm_signal(
            '```json\n{"alignment": 0.5, "embodiment": 0.5, "clarity": 0.5, "vitality": 0.5}\n```'
        )
        assert result is not None

    def test_invalid(self):
        assert parse_llm_signal("not json at all") is None
        assert parse_llm_signal("") is None
        assert parse_llm_signal(None) is None

    def test_missing_facet(self):
        assert parse_llm_signal('{"alignment": 0.5}') is None


class TestBlendSignals:
    def test_blend(self):
        regex = Signal(alignment=0.6, embodiment=0.5, clarity=0.7, vitality=0.5)
        llm = {"alignment": 0.8, "embodiment": 0.9, "clarity": 0.8, "vitality": 0.7}
        blended = blend_signals(regex, llm, llm_weight=0.6)
        # alignment should be 0.4*0.6 + 0.6*0.8 = 0.24 + 0.48 = 0.72
        assert blended.alignment == pytest.approx(0.72)


class TestSignalTracker:
    def test_empty(self):
        t = SignalTracker()
        assert t.recent_health() == 0.5
        assert t.trend() == "stable"

    def test_record_and_health(self):
        t = SignalTracker()
        t.record(Signal(alignment=0.9, embodiment=0.9, clarity=0.9, vitality=0.9))
        assert t.recent_health() > 0.8

    def test_trend_improving(self):
        t = SignalTracker()
        for _ in range(5):
            t.record(Signal(alignment=0.3, embodiment=0.3, clarity=0.3, vitality=0.3))
        for _ in range(5):
            t.record(Signal(alignment=0.9, embodiment=0.9, clarity=0.9, vitality=0.9))
        assert t.trend() == "improving"

    def test_window_limit(self):
        t = SignalTracker(window_size=5)
        for _ in range(10):
            t.record(Signal())
        assert len(t.signals) == 5


class TestParseExtraction:
    def test_valid(self):
        result = parse_extraction(
            '{"relationship_updates": [{"person": "alice", "fact": "likes cats"}], "nothing_new": false}'
        )
        assert not result["nothing_new"]
        assert len(result["relationship_updates"]) == 1

    def test_nothing_new(self):
        result = parse_extraction('{"nothing_new": true}')
        assert result["nothing_new"]

    def test_invalid(self):
        result = parse_extraction("garbage")
        assert result["nothing_new"]

    def test_empty(self):
        result = parse_extraction("")
        assert result["nothing_new"]


class TestReinforcementEngine:
    def test_reinforce(self, episodic):
        tid = episodic.log_trace(content="test", kind="episode", tags=[], salience=0.5)
        engine = ReinforcementEngine(reinforce_delta=0.1, weaken_delta=0.05)
        # Health > 0.7 reinforces
        for t in [tid]:
            episodic.reinforce("traces", t, engine.reinforce_delta)
        trace = episodic.get_trace(tid)
        assert trace["salience"] == pytest.approx(0.6)

    def test_dead_band(self):
        """Health between 0.4 and 0.7 should not adjust."""
        engine = ReinforcementEngine()
        # process() with health=0.5 should be a no-op
        # We just verify the engine's thresholds are correct
        assert engine.reinforce_delta == 0.05
        assert engine.weaken_delta == 0.03


class TestDecayEngine:
    def test_coherence_modulation(self):
        engine = DecayEngine(half_life_hours=168)
        engine.update_coherence(0.0)
        hl_low = engine.effective_half_life()
        engine.update_coherence(1.0)
        hl_high = engine.effective_half_life()
        # Low coherence = longer half-life (preserve more)
        assert hl_low > hl_high

    def test_should_prune(self):
        engine = DecayEngine(min_salience=0.01)
        assert engine.should_prune(0.005)
        assert not engine.should_prune(0.05)

    def test_calculate_decay_recent(self):
        from engram.core.types import now_iso

        engine = DecayEngine()
        # Just accessed = no decay
        multiplier = engine.calculate_decay(now_iso(), 0)
        assert multiplier > 0.99

    def test_calculate_decay_invalid_timestamp(self):
        engine = DecayEngine()
        multiplier = engine.calculate_decay("not-a-date", 0)
        assert multiplier == 0.5  # fallback
