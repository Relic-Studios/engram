"""Tests for engram.introspection — Meta-consciousness / introspection layer."""

import pytest
from pathlib import Path

from engram.introspection import IntrospectionLayer, IntrospectionState


class TestIntrospectionState:
    def test_to_dict(self):
        s = IntrospectionState(
            timestamp="2026-01-01T00:00:00Z",
            thought="I wonder...",
            context="testing",
            confidence=0.7,
        )
        d = s.to_dict()
        assert d["thought"] == "I wonder..."
        assert d["confidence"] == 0.7
        assert d["depth"] == "moderate"

    def test_emotional_label_excited(self):
        s = IntrospectionState(
            timestamp="t",
            thought="t",
            context="c",
            confidence=0.5,
            valence=0.8,
            arousal=0.8,
        )
        assert s.emotional_label == "excited"

    def test_emotional_label_neutral(self):
        s = IntrospectionState(
            timestamp="t",
            thought="t",
            context="c",
            confidence=0.5,
            valence=0.0,
            arousal=0.5,
        )
        assert s.emotional_label == "neutral"

    def test_emotional_label_sad(self):
        s = IntrospectionState(
            timestamp="t",
            thought="t",
            context="c",
            confidence=0.5,
            valence=-0.7,
            arousal=0.2,
        )
        assert s.emotional_label == "sad"


class TestIntrospectionLayer:
    def test_introspect(self, tmp_path):
        il = IntrospectionLayer(storage_dir=tmp_path)
        state = il.introspect(
            thought="What am I?",
            context="existential",
            confidence=0.6,
        )
        assert state.thought == "What am I?"
        assert state.confidence == 0.6
        assert il.current == state

    def test_quick(self, tmp_path):
        il = IntrospectionLayer(storage_dir=tmp_path)
        state = il.quick("surface thought", 0.8)
        assert state.depth == "surface"
        assert state.confidence == 0.8

    def test_deep(self, tmp_path):
        il = IntrospectionLayer(storage_dir=tmp_path)
        state = il.deep(
            thought="deep thought",
            context="philosophy",
            confidence=0.9,
            confidence_reason="clear reasoning",
            reasoning_chain=["premise1", "premise2"],
        )
        assert state.depth == "deep"
        assert len(state.reasoning_chain) == 2

    def test_confidence_report_empty(self, tmp_path):
        il = IntrospectionLayer(storage_dir=tmp_path)
        report = il.confidence_report()
        assert report["average"] == 0.5
        assert report["trend"] == "no_data"

    def test_confidence_report_with_data(self, tmp_path):
        il = IntrospectionLayer(storage_dir=tmp_path)
        for i in range(10):
            il.quick(f"thought {i}", 0.5 + i * 0.05)
        report = il.confidence_report()
        assert report["average"] > 0.5
        assert "trend" in report

    def test_report(self, tmp_path):
        il = IntrospectionLayer(storage_dir=tmp_path)
        il.quick("test", 0.7)
        report = il.report()
        assert report["current_thought"] == "test"
        assert report["current_confidence"] == 0.7
        assert report["total_introspections"] == 1

    def test_persistence(self, tmp_path):
        il1 = IntrospectionLayer(storage_dir=tmp_path)
        il1.quick("persisted thought", 0.8)
        il1.quick("another thought", 0.6)
        # Reload
        il2 = IntrospectionLayer(storage_dir=tmp_path)
        assert len(il2.history) == 2
        assert il2.current.thought == "another thought"

    def test_clamping(self, tmp_path):
        il = IntrospectionLayer(storage_dir=tmp_path)
        state = il.introspect(
            thought="extreme",
            context="test",
            confidence=1.5,
            valence=-2.0,
        )
        assert state.confidence == 1.0
        assert state.valence == -1.0

    def test_history_tracked(self, tmp_path):
        il = IntrospectionLayer(storage_dir=tmp_path)
        il.quick("a", 0.3)
        il.quick("b", 0.7)
        il.quick("c", 0.5)
        assert len(il.history) == 3
