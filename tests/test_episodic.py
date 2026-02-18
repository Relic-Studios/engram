"""Tests for engram.episodic.store."""

import pytest


class TestEpisodicStore:
    def test_log_message(self, episodic):
        msg_id = episodic.log_message(
            person="alice", speaker="alice", content="hello", source="test"
        )
        assert len(msg_id) == 12

    def test_get_messages(self, episodic):
        episodic.log_message(
            person="alice", speaker="alice", content="hi", source="test"
        )
        episodic.log_message(person="bob", speaker="bob", content="yo", source="test")

        msgs = episodic.get_messages(person="alice")
        assert len(msgs) == 1
        assert msgs[0]["person"] == "alice"

    def test_get_recent_messages(self, episodic):
        for i in range(5):
            episodic.log_message(
                person="alice", speaker="alice", content=f"msg {i}", source="test"
            )
        recent = episodic.get_recent_messages("alice", limit=3)
        assert len(recent) == 3
        # Should be chronological
        assert "msg 2" in recent[0]["content"]

    def test_log_trace(self, episodic):
        tid = episodic.log_trace(
            content="Alice mentioned she likes cats",
            kind="episode",
            tags=["alice"],
            salience=0.7,
        )
        assert len(tid) == 12

    def test_get_traces(self, episodic):
        episodic.log_trace(
            content="trace 1", kind="episode", tags=["alice"], salience=0.9
        )
        episodic.log_trace(
            content="trace 2", kind="episode", tags=["bob"], salience=0.3
        )

        traces = episodic.get_traces(min_salience=0.5)
        assert len(traces) == 1
        assert traces[0]["salience"] >= 0.5

    def test_get_traces_by_tag(self, episodic):
        episodic.log_trace(content="alice thing", kind="episode", tags=["alice"])
        episodic.log_trace(content="bob thing", kind="episode", tags=["bob"])

        traces = episodic.get_traces(tags=["alice"])
        assert len(traces) == 1
        assert "alice" in traces[0]["tags"]

    def test_get_by_salience(self, episodic):
        episodic.log_trace(content="low", kind="episode", tags=["alice"], salience=0.1)
        episodic.log_trace(content="high", kind="episode", tags=["alice"], salience=0.9)

        results = episodic.get_by_salience(person="alice", limit=1)
        assert len(results) == 1
        assert results[0]["salience"] == pytest.approx(0.9)

    def test_log_event(self, episodic):
        eid = episodic.log_event(
            type="trust_change", description="promoted alice", person="alice"
        )
        assert len(eid) == 12

    def test_search_messages(self, episodic):
        episodic.log_message(
            person="alice",
            speaker="alice",
            content="I love cats and dogs",
            source="test",
        )
        episodic.log_message(
            person="bob", speaker="bob", content="I like programming", source="test"
        )
        results = episodic.search_messages("cats")
        assert len(results) >= 1
        assert "cats" in results[0]["content"]

    def test_search_traces(self, episodic):
        episodic.log_trace(
            content="learned about python decorators", kind="episode", tags=[]
        )
        results = episodic.search_traces("python decorators")
        assert len(results) >= 1

    def test_reinforce_and_weaken(self, episodic):
        tid = episodic.log_trace(content="test", kind="episode", tags=[], salience=0.5)
        episodic.reinforce("traces", tid, 0.1)
        trace = episodic.get_trace(tid)
        assert trace["salience"] == pytest.approx(0.6)

        episodic.weaken("traces", tid, 0.2)
        trace = episodic.get_trace(tid)
        assert trace["salience"] == pytest.approx(0.4)

    def test_reinforce_clamps(self, episodic):
        tid = episodic.log_trace(content="test", kind="episode", tags=[], salience=0.95)
        episodic.reinforce("traces", tid, 0.2)
        trace = episodic.get_trace(tid)
        assert trace["salience"] <= 1.0

    def test_update_access(self, episodic):
        tid = episodic.log_trace(content="test", kind="episode", tags=[])
        episodic.update_access("traces", tid)
        trace = episodic.get_trace(tid)
        assert trace["access_count"] == 1

    def test_count(self, episodic):
        episodic.log_message(person="alice", speaker="alice", content="x", source="t")
        episodic.log_trace(content="y", kind="episode", tags=[])
        assert episodic.count_messages() == 1
        assert episodic.count_traces() == 1

    def test_prune(self, episodic):
        episodic.log_trace(content="weak", kind="episode", tags=[], salience=0.001)
        episodic.log_trace(content="strong", kind="episode", tags=[], salience=0.9)
        episodic.prune(min_salience=0.01)
        assert episodic.count_traces() == 1

    def test_invalid_table(self, episodic):
        with pytest.raises(ValueError):
            episodic.reinforce("bogus", "id", 0.1)

    def test_context_manager(self, config):
        from engram.episodic.store import EpisodicStore

        with EpisodicStore(config.db_path) as store:
            store.log_message(person="x", speaker="x", content="x", source="x")
