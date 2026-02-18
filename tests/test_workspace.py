"""Tests for engram.workspace — Cognitive workspace (7±2)."""

import pytest
from pathlib import Path

from engram.workspace import CognitiveWorkspace, WorkspaceSlot


class TestWorkspaceSlot:
    def test_defaults(self):
        s = WorkspaceSlot(item="test")
        assert s.priority == 0.5
        assert s.age == 0
        assert s.access_count == 0

    def test_age_step(self):
        s = WorkspaceSlot(item="test", priority=1.0)
        s.age_step(0.9)
        assert s.age == 1
        assert s.priority == pytest.approx(0.9)

    def test_rehearse(self):
        s = WorkspaceSlot(item="test", priority=0.3)
        s.rehearse(0.2)
        assert s.priority == pytest.approx(0.5)
        assert s.age == 0
        assert s.access_count == 1

    def test_is_expired(self):
        s = WorkspaceSlot(item="test", priority=0.05)
        assert s.is_expired(0.1)
        s.priority = 0.2
        assert not s.is_expired(0.1)

    def test_roundtrip(self):
        s = WorkspaceSlot(item="hello", priority=0.7, source="test")
        d = s.to_dict()
        s2 = WorkspaceSlot.from_dict(d)
        assert s2.item == "hello"
        assert s2.priority == pytest.approx(0.7)
        assert s2.source == "test"


class TestCognitiveWorkspace:
    def test_add(self, tmp_path):
        ws = CognitiveWorkspace(capacity=3)
        result = ws.add("item1", priority=0.8)
        assert result["action"] == "added"
        assert len(ws.slots) == 1

    def test_add_dedup_rehearses(self, tmp_path):
        ws = CognitiveWorkspace(capacity=3)
        ws.add("item1", priority=0.5)
        result = ws.add("item1", priority=0.8)
        assert result["action"] == "rehearsed"
        assert len(ws.slots) == 1

    def test_add_evicts_when_full(self):
        ws = CognitiveWorkspace(capacity=2)
        ws.add("low", priority=0.1)
        ws.add("high", priority=0.9)
        result = ws.add("new", priority=0.5)
        assert result["evicted"] is not None
        assert result["evicted"]["item"] == "low"
        assert len(ws.slots) == 2

    def test_eviction_callback(self):
        evicted = []
        ws = CognitiveWorkspace(capacity=2, on_evict=lambda d: evicted.append(d))
        ws.add("low", priority=0.1)
        ws.add("high", priority=0.9)
        ws.add("new", priority=0.5)
        assert len(evicted) == 1
        assert "content" in evicted[0]
        assert evicted[0]["content"] == "low"

    def test_access(self):
        ws = CognitiveWorkspace(capacity=5)
        ws.add("item1", priority=0.5)
        result = ws.access(0)
        assert result == "item1"
        assert ws.slots[0].access_count == 1

    def test_access_out_of_range(self):
        ws = CognitiveWorkspace(capacity=5)
        assert ws.access(99) is None

    def test_find(self):
        ws = CognitiveWorkspace(capacity=5)
        ws.add("apple pie recipe")
        ws.add("banana split")
        idx = ws.find("apple")
        assert idx == 0
        assert ws.find("cherry") is None

    def test_age_step_expires(self):
        ws = CognitiveWorkspace(capacity=5, decay_rate=0.01, expiry_threshold=0.1)
        ws.add("dying", priority=0.2)
        expired = ws.age_step()
        assert expired == 1
        assert len(ws.slots) == 0

    def test_items_sorted(self):
        ws = CognitiveWorkspace(capacity=5)
        ws.add("low", priority=0.2)
        ws.add("high", priority=0.9)
        ws.add("mid", priority=0.5)
        items = ws.items()
        assert items[0] == "high"

    def test_items_limited(self):
        ws = CognitiveWorkspace(capacity=5)
        ws.add("a", priority=0.3)
        ws.add("b", priority=0.8)
        ws.add("c", priority=0.5)
        items = ws.items(n=2)
        assert len(items) == 2

    def test_detailed(self):
        ws = CognitiveWorkspace(capacity=5)
        ws.add("test")
        d = ws.detailed()
        assert len(d) == 1
        assert "item" in d[0]
        assert "priority" in d[0]

    def test_status(self):
        ws = CognitiveWorkspace(capacity=5)
        ws.add("test")
        s = ws.status()
        assert "1/5" in s
        assert "test" in s

    def test_clear(self):
        ws = CognitiveWorkspace(capacity=5)
        ws.add("a")
        ws.add("b")
        ws.clear()
        assert len(ws.slots) == 0

    def test_persistence(self, tmp_path):
        path = tmp_path / "workspace.json"
        ws1 = CognitiveWorkspace(capacity=5, storage_path=path)
        ws1.add("persisted item", priority=0.7)
        # Reload
        ws2 = CognitiveWorkspace(capacity=5, storage_path=path)
        assert len(ws2.slots) == 1
        assert ws2.slots[0].item == "persisted item"

    def test_focus_index(self):
        ws = CognitiveWorkspace(capacity=5)
        ws.add("first")
        assert ws.focus_index == 0
        ws.add("second")
        assert ws.focus_index == 1
        ws.access(0)
        assert ws.focus_index == 0
