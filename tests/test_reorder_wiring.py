"""Tests for D1: Primacy-recency (reorder_u) wiring in server.py and system.py.

Verifies that engram_search, engram_reflect, and boot() all apply
U-shaped reordering so the LLM attends to the most important items
at the start and end of the context window.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from engram.system import MemorySystem
from engram.working.allocator import reorder_u
import engram.server as server_mod


# ── Helpers ──────────────────────────────────────────────────────


def _is_u_shaped(values: list[float]) -> bool:
    """Check that a list of values follows a U-shaped pattern.

    U-shaped means: high at edges, low in the middle.
    For lists <= 2, always True (nothing to check).
    """
    if len(values) <= 2:
        return True
    mid = len(values) // 2
    # The middle value should be <= both the first and last values
    return values[0] >= values[mid] and values[-1] >= values[mid]


def _make_trace(id: str, salience: float, kind: str = "episode") -> dict:
    return {
        "id": id,
        "kind": kind,
        "content": f"Trace {id} content",
        "salience": salience,
        "combined_score": salience,  # mimic search results
    }


# ── reorder_u unit tests (edge cases beyond test_allocator.py) ──


class TestReorderUWithKeyField:
    """Test reorder_u with different key_field values."""

    def test_combined_score_field(self):
        items = [{"id": f"i{i}", "combined_score": float(i) / 10} for i in range(6)]
        result = reorder_u(items, key_field="combined_score")
        scores = [r["combined_score"] for r in result]
        assert _is_u_shaped(scores)

    def test_salience_field(self):
        items = [{"id": f"i{i}", "salience": float(i) / 10} for i in range(6)]
        result = reorder_u(items, key_field="salience")
        sals = [r["salience"] for r in result]
        assert _is_u_shaped(sals)

    def test_preserves_all_items(self):
        items = [{"id": f"i{i}", "combined_score": float(i)} for i in range(7)]
        result = reorder_u(items, key_field="combined_score")
        assert len(result) == len(items)
        assert sorted(r["id"] for r in result) == sorted(i["id"] for i in items)

    def test_highest_at_edges(self):
        """The top-2 items should be at position 0 and position -1."""
        items = [
            {"id": "low1", "salience": 0.1},
            {"id": "low2", "salience": 0.2},
            {"id": "mid", "salience": 0.5},
            {"id": "high1", "salience": 0.9},
            {"id": "high2", "salience": 0.8},
        ]
        result = reorder_u(items, key_field="salience")
        edge_ids = {result[0]["id"], result[-1]["id"]}
        assert edge_ids == {"high1", "high2"}


# ── engram_search wiring ────────────────────────────────────────


@pytest.fixture
def server_system(config_with_soul):
    """Wire up a real MemorySystem and inject it into the server module."""
    cfg = config_with_soul
    system = MemorySystem(config=cfg)

    old_system = server_mod._system
    old_source = server_mod._current_source
    old_person = server_mod._current_person
    server_mod._system = system
    server_mod._current_source = "direct"
    server_mod._current_person = "tester"

    yield system

    server_mod._system = old_system
    server_mod._current_source = old_source
    server_mod._current_person = old_person
    system.close()


class TestSearchReordering:
    """engram_search should apply reorder_u to results."""

    def test_search_calls_reorder_u(self, server_system):
        """Verify reorder_u is invoked on search results."""
        # Insert some traces with varying salience
        ep = server_system.episodic
        for i in range(5):
            ep.log_trace(
                kind="episode",
                content=f"unique_search_test_token_{i}",
                tags=["test"],
                salience=0.1 * (i + 1),
            )

        with patch("engram.server.reorder_u", wraps=reorder_u) as mock_reorder:
            result = server_mod.engram_search(
                query="unique_search_test_token", limit=10
            )
            # reorder_u should have been called with key_field="combined_score"
            mock_reorder.assert_called_once()
            call_kwargs = mock_reorder.call_args
            assert call_kwargs[1].get("key_field") == "combined_score" or (
                len(call_kwargs[0]) > 1 and call_kwargs[0][1] == "combined_score"
            )

    def test_search_results_u_shaped(self, server_system):
        """Search results should have U-shaped combined_score ordering."""
        ep = server_system.episodic
        # Insert enough traces to see U-shaping
        for i in range(6):
            ep.log_trace(
                kind="episode",
                content=f"reorder_search_verification_{i}",
                tags=["test"],
                salience=0.1 * (i + 1),
            )

        raw = server_mod.engram_search(query="reorder_search_verification", limit=10)
        data = json.loads(raw)
        if len(data) >= 3:
            scores = [d.get("combined_score", 0) for d in data]
            assert _is_u_shaped(scores)


# ── engram_reflect wiring ───────────────────────────────────────


class TestReflectReordering:
    """engram_reflect should apply reorder_u to reflection traces."""

    def test_reflect_calls_reorder_u(self, server_system):
        """Verify reorder_u is invoked during reflection."""
        ep = server_system.episodic
        for i in range(5):
            ep.log_trace(
                kind="episode",
                content=f"reflect_reorder_topic_{i}",
                tags=["test"],
                salience=0.1 * (i + 1),
            )

        with patch("engram.server.reorder_u", wraps=reorder_u) as mock_reorder:
            result = server_mod.engram_reflect(
                topic="reflect_reorder_topic", depth="quick"
            )
            # reorder_u should be called at least once (for the traces)
            assert mock_reorder.call_count >= 1
            # Find the call with key_field="salience"
            salience_calls = [
                c for c in mock_reorder.call_args_list if "salience" in str(c)
            ]
            assert len(salience_calls) >= 1

    def test_reflect_traces_u_shaped(self, server_system):
        """Reflected traces should have U-shaped salience ordering."""
        ep = server_system.episodic
        for i in range(8):
            ep.log_trace(
                kind="episode",
                content=f"ushape_reflect_test_{i}",
                tags=["test"],
                salience=0.1 * (i + 1),
            )

        raw = server_mod.engram_reflect(topic="ushape_reflect_test", depth="quick")
        data = json.loads(raw)
        traces = data.get("traces", [])
        if len(traces) >= 3:
            sals = [t.get("salience", 0) for t in traces]
            assert _is_u_shaped(sals)


# ── boot() wiring ───────────────────────────────────────────────


class TestBootReordering:
    """boot() should apply reorder_u to ADRs and top_traces."""

    def test_boot_calls_reorder_u(self, server_system):
        """Verify reorder_u is called during boot for ADRs and top_traces."""
        ep = server_system.episodic
        # Seed some ADRs
        for i in range(4):
            ep.log_trace(
                kind="architecture_decision",
                content=f"ADR {i}: decided to use pattern {i}",
                tags=["adr"],
                salience=0.5 + 0.1 * i,
            )
        # Seed some high-salience traces
        for i in range(6):
            ep.log_trace(
                kind="episode",
                content=f"Important pattern {i}",
                tags=["test"],
                salience=0.3 + 0.1 * i,
            )

        with patch("engram.system.reorder_u", wraps=reorder_u) as mock_reorder:
            result = server_system.boot()
            # Should be called at least twice (once for ADRs, once for top_traces)
            assert mock_reorder.call_count >= 2

    def test_boot_adrs_u_shaped(self, server_system):
        """boot() ADRs should have U-shaped salience."""
        ep = server_system.episodic
        for i in range(5):
            ep.log_trace(
                kind="architecture_decision",
                content=f"ADR ushape {i}: architecture choice {i}",
                tags=["adr"],
                salience=0.3 + 0.12 * i,
            )

        result = server_system.boot()
        adrs = result.get("architecture_decisions", [])
        if len(adrs) >= 3:
            sals = [a.get("salience", 0) for a in adrs]
            assert _is_u_shaped(sals)

    def test_boot_top_memories_u_shaped(self, server_system):
        """boot() top_memories should have U-shaped salience."""
        ep = server_system.episodic
        for i in range(8):
            ep.log_trace(
                kind="episode",
                content=f"Top memory ushape {i}",
                tags=["test"],
                salience=0.1 + 0.1 * i,
            )

        result = server_system.boot()
        memories = result.get("top_memories", [])
        if len(memories) >= 3:
            sals = [m.get("salience", 0) for m in memories]
            assert _is_u_shaped(sals)

    def test_boot_empty_traces_no_crash(self, server_system):
        """boot() with no traces should not crash on reorder_u."""
        result = server_system.boot()
        assert result["architecture_decisions"] == []
        assert result["top_memories"] == []


# ── Integration: reorder_u does not lose data ───────────────────


class TestReorderPreservesData:
    """Ensure reorder_u never drops or duplicates items."""

    @pytest.mark.parametrize("n", [3, 5, 7, 10, 15])
    def test_item_count_preserved(self, n):
        items = [_make_trace(f"t{i}", salience=i * 0.1) for i in range(n)]
        result = reorder_u(items, key_field="salience")
        assert len(result) == n

    @pytest.mark.parametrize("n", [3, 5, 7, 10, 15])
    def test_no_duplicates(self, n):
        items = [_make_trace(f"t{i}", salience=i * 0.1) for i in range(n)]
        result = reorder_u(items, key_field="salience")
        ids = [r["id"] for r in result]
        assert len(set(ids)) == n

    @pytest.mark.parametrize("n", [3, 5, 7, 10, 15])
    def test_same_items_different_order(self, n):
        items = [_make_trace(f"t{i}", salience=i * 0.1) for i in range(n)]
        result = reorder_u(items, key_field="salience")
        assert sorted(result, key=lambda x: x["id"]) == sorted(
            items, key=lambda x: x["id"]
        )
