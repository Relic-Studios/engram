"""Tests for engram.emotional — VAD emotional continuity system."""

import pytest
from pathlib import Path

from engram.emotional import EmotionalEvent, EmotionalSystem


class TestEmotionalEvent:
    def test_to_dict(self):
        evt = EmotionalEvent(
            timestamp="2026-01-01T00:00:00Z",
            description="test event",
            valence_delta=0.3,
            arousal_delta=-0.1,
            dominance_delta=0.0,
            source="test",
            intensity=0.7,
        )
        d = evt.to_dict()
        assert d["description"] == "test event"
        assert d["valence_delta"] == 0.3
        assert d["intensity"] == 0.7


class TestEmotionalSystem:
    def test_init_neutral(self, tmp_path):
        es = EmotionalSystem(storage_dir=tmp_path)
        assert es.valence == 0.0
        assert es.arousal == 0.5
        assert es.dominance == 0.5

    def test_update_positive(self, tmp_path):
        es = EmotionalSystem(storage_dir=tmp_path)
        state = es.update("good news", valence_delta=0.5, intensity=0.8)
        assert state["valence"] > 0.0
        assert "mood" in state

    def test_update_negative(self, tmp_path):
        es = EmotionalSystem(storage_dir=tmp_path)
        state = es.update("bad news", valence_delta=-0.6, intensity=0.9)
        assert state["valence"] < 0.0

    def test_update_clamps(self, tmp_path):
        es = EmotionalSystem(storage_dir=tmp_path)
        es.update("extreme joy", valence_delta=2.0, intensity=1.0)
        assert es.valence <= 1.0
        es.update("extreme sadness", valence_delta=-5.0, intensity=1.0)
        assert es.valence >= -1.0

    def test_current_state(self, tmp_path):
        es = EmotionalSystem(storage_dir=tmp_path)
        state = es.current_state()
        assert "valence" in state
        assert "arousal" in state
        assert "dominance" in state
        assert "mood" in state
        assert "trend" in state

    def test_mood_labels(self, tmp_path):
        es = EmotionalSystem(storage_dir=tmp_path)
        # Neutral
        state = es.current_state()
        assert state["mood"] in ("neutral", "calm", "positive")

    def test_grounding_text(self, tmp_path):
        es = EmotionalSystem(storage_dir=tmp_path)
        text = es.grounding_text()
        assert "emotional state" in text.lower()
        assert "valence" in text

    def test_mood_history(self, tmp_path):
        es = EmotionalSystem(storage_dir=tmp_path)
        es.update("event1", valence_delta=0.1)
        es.update("event2", valence_delta=0.2)
        history = es.mood_history(n=5)
        assert len(history) == 2
        assert history[0]["description"] == "event1"

    def test_persistence(self, tmp_path):
        es1 = EmotionalSystem(storage_dir=tmp_path)
        es1.update("happy event", valence_delta=0.5, intensity=0.8)
        v1 = es1.valence
        # Reload
        es2 = EmotionalSystem(storage_dir=tmp_path)
        assert abs(es2.valence - v1) < 0.01

    def test_trend_stable(self, tmp_path):
        es = EmotionalSystem(storage_dir=tmp_path)
        es.update("a", valence_delta=0.01, intensity=0.5)
        es.update("b", valence_delta=0.01, intensity=0.5)
        state = es.current_state()
        assert state["trend"] in ("stable", "insufficient_data")

    def test_events_tracked(self, tmp_path):
        es = EmotionalSystem(storage_dir=tmp_path)
        es.update("first", valence_delta=0.1)
        es.update("second", valence_delta=0.2)
        assert len(es.events) == 2
        assert es.events[0].description == "first"

    def test_decay_rates_configurable(self, tmp_path):
        es = EmotionalSystem(
            storage_dir=tmp_path,
            valence_decay=0.5,
            arousal_decay=0.5,
            dominance_decay=0.5,
        )
        assert es.valence_decay == 0.5
