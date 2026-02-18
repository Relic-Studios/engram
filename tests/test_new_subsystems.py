"""Tests for subsystem integration: episodic new methods, system.py wiring,
workspace, and pipeline integration."""

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
        episodic.log_trace("high", "episode", ["test"], salience=0.9)
        episodic.log_trace("low", "episode", ["test"], salience=0.1)

        high = episodic.get_traces_by_kind("episode", min_salience=0.5)
        assert len(high) == 1
        assert high[0]["content"] == "high"

    def test_get_traces_with_metadata(self, episodic):
        tid = episodic.log_trace(
            "test", "episode", ["test"], salience=0.5, my_key="my_value"
        )
        trace = episodic.get_trace(tid)
        meta = trace.get("metadata", {})
        assert meta.get("my_key") == "my_value"

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
# System wiring: workspace + after-pipeline workspace aging
# ---------------------------------------------------------------------------


class TestSystemWorkspace:
    def test_workspace_exists(self, config):
        ms = MemorySystem(config=config)
        assert ms.workspace is not None
        assert hasattr(ms.workspace, "add")
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
