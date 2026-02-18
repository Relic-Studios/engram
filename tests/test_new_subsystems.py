"""Tests for new subsystem integration: episodic new methods, server new tools,
system.py wiring, and pipeline integration."""

import json
import pytest

from engram.episodic.store import EpisodicStore
from engram.system import MemorySystem


# ---------------------------------------------------------------------------
# Episodic Store: new methods (C6)
# ---------------------------------------------------------------------------


class TestEpisodicNewMethods:
    def test_get_traces_by_kind(self, episodic):
        episodic.log_trace("temporal data", "temporal", ["test"], salience=0.7)
        episodic.log_trace("utility data", "utility", ["test"], salience=0.6)
        episodic.log_trace("normal data", "episode", ["test"], salience=0.5)

        temporal = episodic.get_traces_by_kind("temporal")
        assert len(temporal) == 1
        assert temporal[0]["kind"] == "temporal"

        utility = episodic.get_traces_by_kind("utility")
        assert len(utility) == 1
        assert utility[0]["kind"] == "utility"

    def test_get_traces_by_kind_min_salience(self, episodic):
        episodic.log_trace("high", "temporal", ["test"], salience=0.8)
        episodic.log_trace("low", "temporal", ["test"], salience=0.2)
        results = episodic.get_traces_by_kind("temporal", min_salience=0.5)
        assert len(results) == 1
        assert results[0]["content"] == "high"

    def test_get_traces_with_metadata(self, episodic):
        episodic.log_trace(
            "decay test",
            "temporal",
            ["test"],
            salience=0.7,
            decay_rate=0.05,
            revival_count=2,
        )
        episodic.log_trace(
            "no meta",
            "episode",
            ["test"],
            salience=0.5,
        )
        results = episodic.get_traces_with_metadata("decay_rate")
        assert len(results) == 1
        meta = results[0].get("metadata", {})
        assert meta.get("decay_rate") == 0.05

    def test_update_trace_metadata(self, episodic):
        tid = episodic.log_trace("test", "episode", ["test"], salience=0.5)
        ok = episodic.update_trace_metadata(tid, "q_value", 0.75)
        assert ok is True
        # Verify
        trace = episodic.get_trace(tid)
        meta = trace.get("metadata", {})
        assert meta.get("q_value") == 0.75

    def test_update_trace_metadata_nonexistent(self, episodic):
        ok = episodic.update_trace_metadata("nonexistent", "key", "val")
        assert ok is False

    def test_update_trace_metadata_merge(self, episodic):
        tid = episodic.log_trace(
            "test",
            "episode",
            ["test"],
            salience=0.5,
            existing_key="existing_value",
        )
        episodic.update_trace_metadata(tid, "new_key", "new_value")
        trace = episodic.get_trace(tid)
        meta = trace.get("metadata", {})
        assert meta.get("existing_key") == "existing_value"
        assert meta.get("new_key") == "new_value"


# ---------------------------------------------------------------------------
# System wiring: new subsystems exist on MemorySystem (C4)
# ---------------------------------------------------------------------------


class TestSystemNewSubsystems:
    def test_personality_exists(self, config):
        ms = MemorySystem(config=config)
        assert ms.personality is not None
        assert hasattr(ms.personality, "grounding_text")
        ms.close()

    def test_emotional_exists(self, config):
        ms = MemorySystem(config=config)
        assert ms.emotional is not None
        assert hasattr(ms.emotional, "current_state")
        ms.close()

    def test_workspace_exists(self, config):
        ms = MemorySystem(config=config)
        assert ms.workspace is not None
        assert hasattr(ms.workspace, "add")
        ms.close()

    def test_introspection_exists(self, config):
        ms = MemorySystem(config=config)
        assert ms.introspection is not None
        assert hasattr(ms.introspection, "quick")
        ms.close()

    def test_boot_sequence_exists(self, config):
        ms = MemorySystem(config=config)
        assert ms.boot_sequence is not None
        assert hasattr(ms.boot_sequence, "generate")
        ms.close()

    def test_identity_loop_exists(self, config):
        ms = MemorySystem(config=config)
        assert ms.identity_loop is not None
        assert hasattr(ms.identity_loop, "assess")
        ms.close()

    def test_mode_manager_exists(self, config):
        ms = MemorySystem(config=config)
        assert ms.mode_manager is not None
        assert hasattr(ms.mode_manager, "set_mode")
        ms.close()

    def test_workspace_eviction_to_episodic(self, config):
        ms = MemorySystem(config=config)
        # Fill workspace beyond capacity
        for i in range(ms.workspace.capacity + 1):
            ms.workspace.add(f"item_{i}", priority=0.1 * (i + 1))
        # Check that an evicted item created an episodic trace
        traces = ms.episodic.get_traces(kind="workspace_eviction")
        assert len(traces) >= 1
        ms.close()

    def test_boot_includes_new_fields(self, config):
        ms = MemorySystem(config=config)
        boot = ms.boot()
        assert "personality" in boot
        assert "emotional_state" in boot
        assert "identity_solidification" in boot
        assert "mode" in boot
        assert "boot_priming" in boot
        ms.close()


# ---------------------------------------------------------------------------
# Server new tools (C5) — using init_system pattern from test_server
# ---------------------------------------------------------------------------


class TestServerNewTools:
    """Test the new MCP server tools via direct function calls."""

    @pytest.fixture(autouse=True)
    def _setup(self, config_with_soul):
        """Set up the server with a test system."""
        from engram import server

        server._system = MemorySystem(config=config_with_soul)
        server._current_person = "tester"
        server._current_source = "direct"
        yield
        if server._system:
            server._system.close()
        server._system = None

    def test_personality_get(self):
        from engram.server import engram_personality_get

        result = json.loads(engram_personality_get())
        assert "core" in result
        assert "dominant_traits" in result

    def test_personality_update(self):
        from engram.server import engram_personality_update

        result = json.loads(engram_personality_update("openness", 0.05, "test reason"))
        assert result["trait"] == "openness"
        assert result["delta"] == 0.05

    def test_emotional_state(self):
        from engram.server import engram_emotional_state

        result = json.loads(engram_emotional_state())
        assert "valence" in result
        assert "mood" in result

    def test_emotional_update(self):
        from engram.server import engram_emotional_update

        result = json.loads(
            engram_emotional_update(
                "feeling good",
                valence_delta=0.3,
                intensity=0.7,
            )
        )
        assert result["valence"] > 0

    def test_workspace_add(self):
        from engram.server import engram_workspace_add

        result = json.loads(engram_workspace_add("think about X"))
        assert result["action"] == "added"

    def test_workspace_status(self):
        from engram.server import engram_workspace_add, engram_workspace_status

        engram_workspace_add("test item")
        status = engram_workspace_status()
        assert "test item" in status

    def test_introspect(self):
        from engram.server import engram_introspect

        result = json.loads(engram_introspect("I wonder...", confidence=0.7))
        assert result["thought"] == "I wonder..."
        assert result["confidence"] == 0.7

    def test_introspection_report(self):
        from engram.server import engram_introspect, engram_introspection_report

        engram_introspect("test thought", confidence=0.8)
        result = json.loads(engram_introspection_report())
        assert result["current_thought"] == "test thought"

    def test_identity_assess(self):
        from engram.server import engram_identity_assess

        result = json.loads(engram_identity_assess("I feel curious."))
        assert result["state"] == "aligned"
        assert "belief_score" in result

    def test_identity_report(self):
        from engram.server import engram_identity_report

        result = json.loads(engram_identity_report())
        assert "status" in result

    def test_mode_get(self):
        from engram.server import engram_mode_get

        result = json.loads(engram_mode_get())
        assert "mode" in result

    def test_mode_set(self):
        from engram.server import engram_mode_set

        result = json.loads(engram_mode_set("active", "test"))
        assert result["changed"] is True
        assert result["new"] == "ACTIVE_CONVERSATION"

    def test_mode_set_crosspost(self):
        from engram.server import engram_mode_set, _get_system

        engram_mode_set("deep_work", "focus time")
        events = _get_system().episodic.get_events(type="mode_change")
        assert len(events) >= 1


# ---------------------------------------------------------------------------
# Pipeline integration: new subsystems wired through before/after
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_before_with_personality(self, config_with_soul):
        ms = MemorySystem(config=config_with_soul)
        ctx = ms.before(person="tester", message="hello")
        # Personality grounding should be in context
        assert "Personality" in ctx.text or "Openness" in ctx.text
        ms.close()

    def test_before_with_emotional(self, config_with_soul):
        ms = MemorySystem(config=config_with_soul)
        # Update emotional state first
        ms.emotional.update("good mood", valence_delta=0.5, intensity=0.8)
        ctx = ms.before(person="tester", message="hello")
        assert "emotional" in ctx.text.lower() or "valence" in ctx.text.lower()
        ms.close()

    def test_after_updates_emotional(self, config_with_soul):
        ms = MemorySystem(config=config_with_soul)
        initial_state = ms.emotional.current_state()
        ms.after(
            person="tester",
            their_message="hello",
            response="Hi! I feel great about this conversation.",
        )
        new_state = ms.emotional.current_state()
        # State should have been updated (events count increased)
        assert len(ms.emotional.events) >= 1
        ms.close()

    def test_after_records_introspection(self, config_with_soul):
        ms = MemorySystem(config=config_with_soul)
        ms.after(
            person="tester",
            their_message="hello",
            response="I think this is meaningful.",
        )
        assert len(ms.introspection.history) >= 1
        ms.close()

    def test_after_runs_identity_loop(self, config_with_soul):
        ms = MemorySystem(config=config_with_soul)
        ms.after(
            person="tester",
            their_message="hello",
            response="I feel curious about this.",
        )
        assert len(ms.identity_loop.belief_scores) >= 1
        ms.close()

    def test_after_ages_workspace(self, config_with_soul):
        ms = MemorySystem(config=config_with_soul)
        ms.workspace.add("test item", priority=0.5)
        initial_age = ms.workspace.slots[0].age
        ms.after(
            person="tester",
            their_message="hello",
            response="response",
        )
        assert ms.workspace.slots[0].age == initial_age + 1
        ms.close()
