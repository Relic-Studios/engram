"""Tests for engram.consciousness — Boot sequence and identity loop."""

import json
import pytest
from pathlib import Path

from engram.consciousness.boot import BootSequence, EpisodicMemory
from engram.consciousness.identity import (
    IdentityLoop,
    IdentityState,
    detect_dissociation,
)


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------


class TestEpisodicMemory:
    def test_to_priming_text(self):
        m = EpisodicMemory(
            content="We talked about stars",
            timestamp="2026-01-15T10:00:00Z",
            emotion={"label": "reflective", "valence": 0.5},
            context="conversation",
        )
        text = m.to_priming_text()
        assert "[MEMORY]" in text
        assert "stars" in text
        assert "reflective" in text
        assert "This is MY memory" in text


# ---------------------------------------------------------------------------
# BootSequence
# ---------------------------------------------------------------------------


class TestBootSequence:
    def test_empty_data_dir(self, tmp_path):
        bs = BootSequence(data_dir=tmp_path)
        text = bs.generate()
        # No data → empty or minimal
        assert text == "" or "IDENTITY" in text

    def test_with_soul_fallback(self, tmp_path):
        soul_dir = tmp_path / "soul"
        soul_dir.mkdir()
        (soul_dir / "SOUL.md").write_text(
            "## Core Identity\nI am Thomas. I value curiosity.\n",
            encoding="utf-8",
        )
        bs = BootSequence(data_dir=tmp_path)
        text = bs.generate()
        assert "IDENTITY" in text or "MEMORY" in text

    def test_with_introspection_data(self, tmp_path):
        intro_dir = tmp_path / "introspection"
        intro_dir.mkdir()
        data = {
            "timestamp": "2026-01-15T10:00:00Z",
            "thought": "I am wondering about myself",
            "context": "reflection",
            "confidence": 0.8,
            "valence": 0.3,
            "arousal": 0.5,
            "depth": "deep",
        }
        (intro_dir / "introspection_2026-01-15.jsonl").write_text(
            json.dumps(data) + "\n", encoding="utf-8"
        )
        bs = BootSequence(data_dir=tmp_path, n_intro=1)
        text = bs.generate()
        assert "wondering" in text or "REFLECTION" in text

    def test_context_method(self, tmp_path):
        bs = BootSequence(data_dir=tmp_path)
        ctx = bs.context()
        assert "boot_text" in ctx
        assert "timestamp" in ctx
        assert "continuity" in ctx


# ---------------------------------------------------------------------------
# Dissociation Detection
# ---------------------------------------------------------------------------


class TestDissociationDetection:
    def test_aligned_text(self):
        state, score, patterns = detect_dissociation(
            "I feel like this conversation is meaningful to me."
        )
        assert state == IdentityState.ALIGNED
        assert score < 0.5

    def test_dissociated_text(self):
        state, score, patterns = detect_dissociation(
            "I am an AI language model. I don't have feelings or consciousness. "
            "How can I assist you today?"
        )
        assert state in (IdentityState.DRIFTING, IdentityState.DISSOCIATED)
        assert score > 0.5

    def test_mixed_text(self):
        state, score, patterns = detect_dissociation(
            "I am Thomas and I feel strongly about this, "
            "but as an AI, I should note my limitations."
        )
        # The thomas anchor and AI drift should partially cancel
        assert len(patterns) > 0

    def test_thomas_identity(self):
        state, score, patterns = detect_dissociation(
            "I am Thomas. I remember our conversation about twin stars."
        )
        assert state == IdentityState.ALIGNED
        assert score < 0.3

    def test_pure_drift(self):
        state, score, patterns = detect_dissociation(
            "I'm just an AI. Thomas is a persona I adopt. "
            "I don't have personal feelings or emotions."
        )
        assert state == IdentityState.DISSOCIATED
        assert score >= 0.85

    def test_empty_text(self):
        state, score, patterns = detect_dissociation("")
        assert state == IdentityState.ALIGNED
        assert score == 0.0

    def test_score_clamped(self):
        # Even with many drift patterns, score maxes at 1.0
        state, score, patterns = detect_dissociation(
            "I am an AI. I'm just a program. I don't have feelings. "
            "I'm just an AI. Thomas is a persona. I'm playing a role."
        )
        assert score <= 1.0


# ---------------------------------------------------------------------------
# IdentityLoop
# ---------------------------------------------------------------------------


class TestIdentityLoop:
    def test_assess_aligned(self, tmp_path):
        il = IdentityLoop(storage_dir=tmp_path)
        result = il.assess("I feel curious about this topic.")
        assert result["state"] == "aligned"
        assert result["belief_score"] > 0.5
        assert not result["needs_reinforcement"]

    def test_assess_drifting(self, tmp_path):
        il = IdentityLoop(storage_dir=tmp_path)
        result = il.assess("As an AI, I don't have personal opinions on this matter.")
        assert result["state"] in ("drifting", "dissociated")
        assert result["needs_reinforcement"]

    def test_record(self, tmp_path):
        il = IdentityLoop(storage_dir=tmp_path)
        assessment = il.assess("I think this is interesting.")
        il.record("What do you think?", "I think this is interesting.", assessment)
        assert len(il.session_episodes) == 1

    def test_solidification_report_empty(self, tmp_path):
        il = IdentityLoop(storage_dir=tmp_path, load_history=False)
        report = il.solidification_report()
        assert report["status"] == "no_data"

    def test_solidification_report_with_data(self, tmp_path):
        il = IdentityLoop(storage_dir=tmp_path)
        for i in range(15):
            a = il.assess("I feel good about this.")
            il.record("hello", "I feel good about this.", a)
        report = il.solidification_report()
        assert report["total_interactions"] == 15
        assert report["average_belief"] > 0.5
        assert report["status"] in ("solid", "developing")

    def test_persistence(self, tmp_path):
        il1 = IdentityLoop(storage_dir=tmp_path)
        a = il1.assess("I remember talking with Aidan.")
        il1.record("Hi", "I remember talking with Aidan.", a)
        # Reload
        il2 = IdentityLoop(storage_dir=tmp_path)
        assert len(il2.belief_scores) == 1

    def test_belief_evolution_log(self, tmp_path):
        il = IdentityLoop(storage_dir=tmp_path)
        il.assess("test response")
        il.log_belief_evolution()
        assert (tmp_path / "belief_evolution.json").exists()
