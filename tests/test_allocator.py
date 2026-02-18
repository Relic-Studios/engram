"""Tests for engram.working.allocator — knapsack + MMR diversity."""

import pytest

from engram.working.allocator import (
    knapsack_allocate,
    reorder_u,
    _cosine_similarity,
    _max_sim_to_selected,
)


# ── Helpers ──────────────────────────────────────────────────────


def _item(id: str, content: str, salience: float = 0.5) -> dict:
    return {"id": id, "content": content, "salience": salience}


# ── Cosine similarity ───────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


class TestMaxSimToSelected:
    def test_empty_selected(self):
        assert _max_sim_to_selected([1.0, 2.0], []) == 0.0

    def test_returns_max(self):
        candidate = [1.0, 0.0]
        selected = [[1.0, 0.0], [0.0, 1.0]]
        assert _max_sim_to_selected(candidate, selected) == pytest.approx(1.0)


# ── Knapsack (no MMR) ───────────────────────────────────────────


class TestKnapsackBasic:
    def test_empty(self):
        selected, used = knapsack_allocate([], 1000)
        assert selected == []
        assert used == 0

    def test_zero_budget(self):
        items = [_item("a", "hello world", 0.9)]
        selected, used = knapsack_allocate(items, 0)
        assert selected == []

    def test_selects_by_density(self):
        items = [
            _item("a", "short", 0.9),  # high density
            _item("b", "x " * 200, 0.91),  # low density (many tokens)
            _item("c", "medium len text", 0.5),
        ]
        selected, used = knapsack_allocate(items, 100)
        ids = [s["id"] for s in selected]
        assert "a" in ids  # highest density should be selected

    def test_budget_respected(self):
        items = [_item(f"i{i}", f"word{i} " * 50, 0.5) for i in range(20)]
        selected, used = knapsack_allocate(items, 50)
        assert used <= 50


# ── Knapsack with MMR ────────────────────────────────────────────


class TestKnapsackMMR:
    def test_no_embeddings_fallback(self):
        """Without embeddings, MMR is not applied (greedy fallback)."""
        items = [_item("a", "hello", 0.9), _item("b", "world", 0.8)]
        sel1, _ = knapsack_allocate(items, 1000)
        sel2, _ = knapsack_allocate(items, 1000, embeddings=None)
        assert [s["id"] for s in sel1] == [s["id"] for s in sel2]

    def test_mmr_penalises_duplicates(self):
        """Near-duplicate embeddings should be penalised by MMR."""
        # All items have same token count so density = salience
        items = [
            _item("dup1", "word", 0.9),
            _item("dup2", "word", 0.85),
            _item("diff", "word", 0.8),
        ]
        embeddings = {
            "dup1": [1.0, 0.0, 0.0],
            "dup2": [0.99, 0.01, 0.0],  # nearly identical to dup1
            "diff": [0.0, 0.0, 1.0],  # orthogonal
        }
        selected, _ = knapsack_allocate(
            items, 1000, embeddings=embeddings, mmr_lambda=0.5
        )
        ids = [s["id"] for s in selected]
        # After selecting dup1, MMR should prefer "diff" over "dup2"
        assert ids[0] == "dup1"
        assert ids[1] == "diff"  # diversity wins over dup2's higher salience
        assert ids[2] == "dup2"

    def test_mmr_lambda_1_equals_greedy(self):
        """mmr_lambda=1.0 should behave identically to greedy."""
        items = [
            _item("a", "hello", 0.9),
            _item("b", "world", 0.8),
            _item("c", "test", 0.7),
        ]
        embeddings = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]}
        sel_greedy, _ = knapsack_allocate(items, 1000)
        sel_mmr, _ = knapsack_allocate(
            items, 1000, embeddings=embeddings, mmr_lambda=1.0
        )
        assert [s["id"] for s in sel_greedy] == [s["id"] for s in sel_mmr]

    def test_mmr_with_missing_embeddings(self):
        """Items without embeddings should still be selected (graceful)."""
        items = [
            _item("a", "hello", 0.9),
            _item("b", "world", 0.8),
        ]
        embeddings = {"a": [1.0, 0.0]}  # only one item has embedding
        selected, _ = knapsack_allocate(
            items, 1000, embeddings=embeddings, mmr_lambda=0.7
        )
        assert len(selected) == 2


# ── Reorder U ────────────────────────────────────────────────────


class TestReorderU:
    def test_small_list_unchanged(self):
        items = [_item("a", "x", 0.9)]
        assert reorder_u(items) == items

    def test_two_items_unchanged(self):
        items = [_item("a", "x", 0.9), _item("b", "y", 0.8)]
        assert reorder_u(items) == items

    def test_u_shape_ordering(self):
        items = [_item(f"i{i}", "x", float(i)) for i in range(5)]
        result = reorder_u(items)
        # Highest at start and end, lowest in middle
        sals = [r["salience"] for r in result]
        assert sals[0] >= sals[2]  # start > middle
        assert sals[-1] >= sals[2]  # end > middle
