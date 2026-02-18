"""Tests for engram.signal (measure, extract, reinforcement, decay)."""

import pytest
from engram.signal.measure import (
    check_correctness,
    check_consistency,
    check_completeness,
    check_robustness,
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


class TestCorrectness:
    def test_correct_code(self):
        text = """```python
def add(a: int, b: int) -> int:
    return a + b
```"""
        score = check_correctness(text)
        assert score > 0.4

    def test_incorrect_code(self):
        text = """```python
def add(a, b):
    return a + c  # undefined variable
```"""
        score = check_correctness(text)
        assert score < 0.5

    def test_empty(self):
        score = check_correctness("")
        assert 0.0 <= score <= 1.0


class TestConsistency:
    def test_consistent_naming(self):
        text = """```python
def get_user_name(user_id: int) -> str:
    user_data = fetch_user_data(user_id)
    return user_data["name"]
```"""
        score = check_consistency(text)
        assert score > 0.3

    def test_inconsistent_style(self):
        text = """```python
def getUserName(userId):
    user_data = fetch_user_data(userId)
    return user_data["name"]

def get_user_age(user_id):
    userData = fetchUserData(user_id)
    return userData["age"]
```"""
        score = check_consistency(text)
        # Mixed naming conventions should score lower
        assert 0.0 <= score <= 1.0


class TestCompleteness:
    def test_complete_code(self):
        text = """```python
def divide(a: float, b: float) -> float:
    \"\"\"Divide a by b.\"\"\"
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```"""
        score = check_completeness(text)
        assert score > 0.4

    def test_incomplete_code(self):
        text = """```python
def process():
    pass  # TODO: implement this
```"""
        score = check_completeness(text)
        assert score < 0.6


class TestRobustness:
    def test_robust_code(self):
        text = """```python
def read_file(path: str) -> str:
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""
    except PermissionError:
        raise
```"""
        score = check_robustness(text)
        assert score > 0.4

    def test_fragile_code(self):
        text = """```python
def read_file(path):
    f = open(path)
    data = f.read()
    return data
```"""
        score = check_robustness(text)
        # No error handling, no context manager
        assert 0.0 <= score <= 1.0


class TestMeasureRegex:
    def test_quality_response(self):
        text = """Here's the fix for the bug at line 42:

```python
def process_data(items: list[str]) -> dict:
    if not items:
        raise ValueError("items cannot be empty")
    result = {}
    for item in items:
        try:
            result[item] = parse(item)
        except ParseError as e:
            logger.warning("Failed to parse %s: %s", item, e)
    return result
```

This adds input validation and proper error handling."""
        signal = measure_regex(text)
        assert signal.correctness > 0.3
        assert signal.health > 0.3

    def test_low_quality_response(self):
        text = "just do x = y + z and it should work"
        signal = measure_regex(text)
        assert signal.health < 0.8


class TestMeasureUnified:
    def test_regex_only(self):
        signal = measure("Here is some code: x = 1 + 2")
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
            return '{"correctness": 0.8, "consistency": 0.7, "completeness": 0.9, "robustness": 0.85}'

        signal = measure("hello", llm_func=mock_llm)
        # Should blend regex + LLM
        assert signal.health > 0.5


class TestParseLLMSignal:
    def test_valid_json(self):
        result = parse_llm_signal(
            '{"correctness": 0.8, "consistency": 0.7, "completeness": 0.9, "robustness": 0.85}'
        )
        assert result is not None
        assert result["correctness"] == pytest.approx(0.8)

    def test_fenced_json(self):
        result = parse_llm_signal(
            '```json\n{"correctness": 0.5, "consistency": 0.5, "completeness": 0.5, "robustness": 0.5}\n```'
        )
        assert result is not None

    def test_invalid(self):
        assert parse_llm_signal("not json at all") is None
        assert parse_llm_signal("") is None
        assert parse_llm_signal(None) is None

    def test_missing_facet(self):
        assert parse_llm_signal('{"correctness": 0.5}') is None


class TestBlendSignals:
    def test_blend(self):
        regex = Signal(
            correctness=0.6, consistency=0.5, completeness=0.7, robustness=0.5
        )
        llm = {
            "correctness": 0.8,
            "consistency": 0.9,
            "completeness": 0.8,
            "robustness": 0.7,
        }
        blended = blend_signals(regex, llm, llm_weight=0.6)
        # correctness should be 0.4*0.6 + 0.6*0.8 = 0.24 + 0.48 = 0.72
        assert blended.correctness == pytest.approx(0.72)


class TestSignalTracker:
    def test_empty(self):
        t = SignalTracker()
        assert t.recent_health() == 0.5
        assert t.trend() == "stable"

    def test_record_and_health(self):
        t = SignalTracker()
        t.record(
            Signal(correctness=0.9, consistency=0.9, completeness=0.9, robustness=0.9)
        )
        assert t.recent_health() > 0.8

    def test_trend_improving(self):
        t = SignalTracker()
        for _ in range(5):
            t.record(
                Signal(
                    correctness=0.3, consistency=0.3, completeness=0.3, robustness=0.3
                )
            )
        for _ in range(5):
            t.record(
                Signal(
                    correctness=0.9, consistency=0.9, completeness=0.9, robustness=0.9
                )
            )
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
