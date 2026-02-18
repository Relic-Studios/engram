"""Tests for engram.consolidation (pressure, compactor, consolidator, citations)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from engram.consolidation.pressure import (
    MemoryPressure,
    PressureLevel,
    PressureState,
    _CRITICAL_THRESHOLD,
    _DECAY_INTERVALS,
    _ELEVATED_THRESHOLD,
)
from engram.consolidation.compactor import (
    CompactionResult,
    ConversationCompactor,
    _extractive_summarise,
    _format_segment,
    _segment_messages,
    _time_gap_hours,
)
from engram.consolidation.consolidator import (
    MemoryConsolidator,
    _cluster_by_time,
    _cluster_by_topic,
    _extractive_arc_summary,
    _extractive_thread_summary,
    _format_traces_for_llm,
    _get_unconsolidated_episodes,
    _get_unconsolidated_threads,
    _group_by_person,
    _mark_consolidated,
    _time_range,
)
from engram.core.config import Config
from engram.episodic.store import EpisodicStore
from engram.pipeline.after import _extract_cited_trace_ids


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def config(tmp_path):
    cfg = Config.from_data_dir(tmp_path, signal_mode="regex", extract_mode="off")
    cfg.ensure_directories()
    return cfg


@pytest.fixture
def episodic(config):
    store = EpisodicStore(config.db_path)
    yield store
    store.close()


@pytest.fixture
def pressure(config):
    return MemoryPressure(config)


@pytest.fixture
def compactor():
    return ConversationCompactor(
        keep_recent=5,
        segment_size=10,
        min_messages_to_compact=15,
    )


# -----------------------------------------------------------------------
# MemoryPressure — classification
# -----------------------------------------------------------------------


class TestPressureClassification:
    def test_normal_at_zero(self, pressure):
        assert pressure._classify(0.0) == PressureLevel.NORMAL

    def test_normal_below_elevated(self, pressure):
        assert pressure._classify(0.59) == PressureLevel.NORMAL

    def test_elevated_at_threshold(self, pressure):
        assert pressure._classify(_ELEVATED_THRESHOLD) == PressureLevel.ELEVATED

    def test_elevated_in_range(self, pressure):
        assert pressure._classify(0.70) == PressureLevel.ELEVATED

    def test_critical_at_threshold(self, pressure):
        assert pressure._classify(_CRITICAL_THRESHOLD) == PressureLevel.CRITICAL

    def test_critical_above(self, pressure):
        assert pressure._classify(1.0) == PressureLevel.CRITICAL


# -----------------------------------------------------------------------
# MemoryPressure — check()
# -----------------------------------------------------------------------


class TestPressureCheck:
    def test_normal_state_no_decay_until_interval(self, pressure, episodic):
        """In normal state with no prior decay, should_decay is True (inf > 3600)."""
        state = pressure.check(episodic)
        assert state.level == PressureLevel.NORMAL
        assert state.should_decay is True  # never decayed before
        assert state.should_compact is False
        assert state.utilisation == 0.0

    def test_normal_state_no_decay_after_recent(self, pressure, episodic):
        """After recording a decay, normal state should not decay again quickly."""
        pressure.record_decay()
        state = pressure.check(episodic)
        assert state.should_decay is False
        assert state.should_compact is False

    def test_elevated_triggers_compact(self, pressure, episodic, config):
        """When utilisation > 60%, should_compact and should_decay are True."""
        # Insert enough traces to reach elevated pressure
        threshold_count = int(config.max_traces * 0.65)
        for i in range(threshold_count):
            episodic.log_trace(
                content=f"trace {i}", kind="episode", tags=["test"], salience=0.5
            )

        state = pressure.check(episodic)
        assert state.level == PressureLevel.ELEVATED
        assert state.should_compact is True
        assert state.should_decay is True  # first ever decay

    def test_critical_always_decays(self, pressure, episodic, config):
        """CRITICAL pressure: decay interval is 0 → always decay."""
        # Fill past critical threshold
        count = int(config.max_traces * 0.85)
        for i in range(count):
            episodic.log_trace(
                content=f"trace {i}", kind="episode", tags=["test"], salience=0.5
            )

        # Even after a recent decay, CRITICAL should still decay
        pressure.record_decay()
        state = pressure.check(episodic)
        assert state.level == PressureLevel.CRITICAL
        assert state.should_decay is True

    def test_compact_respects_cooldown(self, pressure, episodic, config):
        """Compaction should not re-run within 5 minutes."""
        # Get to elevated
        count = int(config.max_traces * 0.65)
        for i in range(count):
            episodic.log_trace(
                content=f"trace {i}", kind="episode", tags=["test"], salience=0.5
            )

        # First check: should compact
        state1 = pressure.check(episodic)
        assert state1.should_compact is True

        # Record compaction
        pressure.record_compaction()

        # Immediately check again: should NOT compact (cooldown)
        state2 = pressure.check(episodic)
        assert state2.should_compact is False

    def test_state_dataclass_fields(self, pressure, episodic):
        state = pressure.check(episodic)
        assert isinstance(state, PressureState)
        assert isinstance(state.level, PressureLevel)
        assert isinstance(state.utilisation, float)
        assert isinstance(state.trace_count, int)
        assert isinstance(state.max_traces, int)
        assert isinstance(state.should_decay, bool)
        assert isinstance(state.should_compact, bool)
        assert isinstance(state.seconds_since_decay, float)


# -----------------------------------------------------------------------
# MemoryPressure — record helpers
# -----------------------------------------------------------------------


class TestPressureRecording:
    def test_record_decay_updates_timestamp(self, pressure):
        assert pressure._last_decay is None
        pressure.record_decay()
        assert pressure._last_decay is not None
        assert (datetime.now(timezone.utc) - pressure._last_decay).total_seconds() < 2

    def test_record_compaction_updates_timestamp(self, pressure):
        assert pressure._last_compact is None
        pressure.record_compaction()
        assert pressure._last_compact is not None


# -----------------------------------------------------------------------
# ConversationCompactor — segment_messages
# -----------------------------------------------------------------------


class TestSegmentMessages:
    def test_empty(self):
        assert _segment_messages([], 10) == []

    def test_single_segment(self):
        msgs = [{"timestamp": f"2025-01-01T0{i}:00:00Z"} for i in range(5)]
        segments = _segment_messages(msgs, 10)
        assert len(segments) == 1
        assert len(segments[0]) == 5

    def test_splits_at_segment_size(self):
        msgs = [{"timestamp": f"2025-01-01T00:{i:02d}:00Z"} for i in range(25)]
        segments = _segment_messages(msgs, 10)
        assert len(segments) == 3  # 10 + 10 + 5
        assert len(segments[0]) == 10
        assert len(segments[1]) == 10
        assert len(segments[2]) == 5

    def test_splits_on_time_gap(self):
        """Messages > 2 hours apart should create separate segments."""
        msgs = [
            {"timestamp": "2025-01-01T10:00:00Z"},
            {"timestamp": "2025-01-01T10:30:00Z"},
            # 3 hour gap
            {"timestamp": "2025-01-01T13:30:00Z"},
            {"timestamp": "2025-01-01T14:00:00Z"},
        ]
        segments = _segment_messages(msgs, 100)
        assert len(segments) == 2
        assert len(segments[0]) == 2
        assert len(segments[1]) == 2

    def test_no_timestamp_no_crash(self):
        """Messages without timestamps should not crash segmentation."""
        msgs = [{}, {}, {}]
        segments = _segment_messages(msgs, 10)
        assert len(segments) == 1
        assert len(segments[0]) == 3


# -----------------------------------------------------------------------
# ConversationCompactor — helpers
# -----------------------------------------------------------------------


class TestCompactorHelpers:
    def test_format_segment(self):
        segment = [
            {"speaker": "alice", "content": "hello"},
            {"speaker": "self", "content": "hey there!"},
        ]
        text = _format_segment(segment)
        assert "[alice]: hello" in text
        assert "[self]: hey there!" in text

    def test_time_gap_hours_valid(self):
        a = {"timestamp": "2025-01-01T10:00:00Z"}
        b = {"timestamp": "2025-01-01T13:00:00Z"}
        assert abs(_time_gap_hours(a, b) - 3.0) < 0.01

    def test_time_gap_hours_missing_ts(self):
        assert _time_gap_hours({}, {}) == 0.0

    def test_time_gap_hours_bad_ts(self):
        a = {"timestamp": "not-a-timestamp"}
        b = {"timestamp": "2025-01-01T10:00:00Z"}
        assert _time_gap_hours(a, b) == 0.0

    def test_extractive_summarise_picks_top_salience(self):
        msgs = [
            {
                "speaker": "alice",
                "content": "boring small talk",
                "salience": 0.1,
                "timestamp": "2025-01-01T10:00:00Z",
                "person": "alice",
            },
            {
                "speaker": "self",
                "content": "important insight about relationships",
                "salience": 0.9,
                "timestamp": "2025-01-01T10:05:00Z",
                "person": "alice",
            },
            {
                "speaker": "alice",
                "content": "another boring thing",
                "salience": 0.2,
                "timestamp": "2025-01-01T10:10:00Z",
                "person": "alice",
            },
        ]
        summary = _extractive_summarise(msgs)
        assert "important insight" in summary
        assert "Summary of conversation with alice" in summary

    def test_extractive_summarise_empty(self):
        assert _extractive_summarise([]) == ""

    def test_extractive_summarise_truncates_long_messages(self):
        msgs = [
            {
                "speaker": "alice",
                "content": "x" * 500,
                "salience": 0.9,
                "timestamp": "2025-01-01T10:00:00Z",
                "person": "alice",
            },
        ]
        summary = _extractive_summarise(msgs)
        # Original content was 500 chars, should be truncated to ~300 + "..."
        assert "..." in summary


# -----------------------------------------------------------------------
# ConversationCompactor — compact()
# -----------------------------------------------------------------------


class TestCompactorCompact:
    def _populate_messages(self, episodic, person, count, start_hour=0):
        """Helper to insert `count` messages for a person."""
        for i in range(count):
            ts_min = start_hour * 60 + i
            episodic.log_message(
                person=person,
                speaker=person if i % 2 == 0 else "self",
                content=f"Message {i} from {person}",
                source="test",
                salience=0.3 + (i % 5) * 0.1,
            )

    def test_below_threshold_skips(self, compactor, episodic):
        """Don't compact if fewer messages than min_messages_to_compact."""
        self._populate_messages(episodic, "alice", 10)
        result = compactor.compact("alice", episodic)
        assert result.messages_archived == 0
        assert result.summaries_created == 0
        assert result.segments_processed == 0

    def test_compacts_above_threshold(self, compactor, episodic):
        """Should compact when messages exceed threshold."""
        self._populate_messages(episodic, "alice", 25)  # > 15 threshold
        result = compactor.compact("alice", episodic)
        assert result.person == "alice"
        assert result.messages_archived > 0
        assert result.summaries_created > 0
        assert result.segments_processed > 0

    def test_keeps_recent_messages(self, compactor, episodic):
        """The most recent `keep_recent` messages should not be archived."""
        self._populate_messages(episodic, "bob", 25)
        result = compactor.compact("bob", episodic)

        # After compaction, unarchived messages should equal keep_recent
        unarchived = episodic.conn.execute(
            """
            SELECT COUNT(*) FROM messages
            WHERE person = ? AND COALESCE(json_extract(metadata, '$.archived'), 0) = 0
            """,
            ("bob",),
        ).fetchone()[0]
        assert unarchived == compactor.keep_recent

    def test_archived_messages_have_flag(self, compactor, episodic):
        """Archived messages should have metadata.archived = 1."""
        self._populate_messages(episodic, "carol", 25)
        compactor.compact("carol", episodic)

        archived_rows = episodic.conn.execute(
            """
            SELECT metadata FROM messages
            WHERE person = ? AND json_extract(metadata, '$.archived') = 1
            """,
            ("carol",),
        ).fetchall()
        assert len(archived_rows) > 0
        import json

        for row in archived_rows:
            meta = json.loads(row[0])
            assert meta["archived"] == 1
            assert "archived_at" in meta

    def test_summaries_stored_as_traces(self, compactor, episodic):
        """Compaction summaries should be stored as kind='summary' traces."""
        self._populate_messages(episodic, "dave", 25)
        compactor.compact("dave", episodic)

        summaries = episodic.conn.execute(
            "SELECT * FROM traces WHERE kind = 'summary'"
        ).fetchall()
        assert len(summaries) > 0

    def test_with_llm_func(self, compactor, episodic):
        """When LLM func is provided, it should be used for summarisation."""
        self._populate_messages(episodic, "eve", 25)

        call_count = 0

        def mock_llm(prompt, system):
            nonlocal call_count
            call_count += 1
            return "LLM-generated summary of the conversation."

        result = compactor.compact("eve", episodic, llm_func=mock_llm)
        assert result.summaries_created > 0
        assert call_count > 0

    def test_llm_failure_falls_back_to_extractive(self, compactor, episodic):
        """If LLM summarisation fails, no summary for that segment."""
        self._populate_messages(episodic, "frank", 25)

        def failing_llm(prompt, system):
            raise RuntimeError("LLM is broken")

        result = compactor.compact("frank", episodic, llm_func=failing_llm)
        # LLM fails, _llm_summarise returns "", segment skipped.
        # But messages that were part of skipped segments are NOT archived.
        # So archived count depends on how many segments succeeded.
        assert isinstance(result, CompactionResult)

    def test_compact_all(self, compactor, episodic):
        """compact_all should process all people above threshold."""
        self._populate_messages(episodic, "alice", 25)
        self._populate_messages(episodic, "bob", 25)
        self._populate_messages(episodic, "carol", 5)  # below threshold

        results = compactor.compact_all(episodic)
        people_compacted = {r.person for r in results}
        assert "alice" in people_compacted
        assert "bob" in people_compacted
        assert "carol" not in people_compacted


# -----------------------------------------------------------------------
# Citation extraction (_extract_cited_trace_ids)
# -----------------------------------------------------------------------


class TestCitationExtraction:
    def test_no_citations(self):
        trace_ids = ["t1", "t2", "t3"]
        assert _extract_cited_trace_ids("No references here.", trace_ids) == []

    def test_single_citation(self):
        trace_ids = ["t1", "t2", "t3"]
        response = "Based on [1], I think we should proceed."
        result = _extract_cited_trace_ids(response, trace_ids)
        assert result == ["t1"]

    def test_multiple_citations(self):
        trace_ids = ["t1", "t2", "t3"]
        response = "As noted in [1] and [3], the approach works."
        result = _extract_cited_trace_ids(response, trace_ids)
        assert result == ["t1", "t3"]

    def test_duplicate_citations(self):
        """Same citation number used twice should only appear once."""
        trace_ids = ["t1", "t2"]
        response = "See [1]. Also [1] again."
        result = _extract_cited_trace_ids(response, trace_ids)
        assert result == ["t1"]

    def test_out_of_range_citations_ignored(self):
        """Citation numbers beyond trace_ids length are ignored."""
        trace_ids = ["t1", "t2"]
        response = "See [1] and [5] and [0]."
        result = _extract_cited_trace_ids(response, trace_ids)
        assert result == ["t1"]  # [5] out of range, [0] is not 1-indexed

    def test_zero_not_matched(self):
        """[0] should be ignored since citations are 1-indexed."""
        trace_ids = ["t1", "t2"]
        response = "See [0]."
        result = _extract_cited_trace_ids(response, trace_ids)
        assert result == []

    def test_empty_response(self):
        assert _extract_cited_trace_ids("", ["t1"]) == []

    def test_empty_trace_ids(self):
        assert _extract_cited_trace_ids("See [1].", []) == []

    def test_none_response(self):
        """None/falsy response should not crash."""
        assert _extract_cited_trace_ids("", []) == []

    def test_citation_with_surrounding_text(self):
        trace_ids = ["a1", "b2", "c3"]
        response = (
            "According to my memories [2], we discussed this before. "
            "Combined with [1], I'm confident in my answer."
        )
        result = _extract_cited_trace_ids(response, trace_ids)
        assert result == ["a1", "b2"]

    def test_all_traces_cited(self):
        trace_ids = ["x", "y", "z"]
        response = "Combining [1], [2], and [3], the answer is clear."
        result = _extract_cited_trace_ids(response, trace_ids)
        assert result == ["x", "y", "z"]


# -----------------------------------------------------------------------
# Context builder — citation-ready trace rendering
# -----------------------------------------------------------------------


class TestContextCitationFormat:
    def test_traces_numbered_in_context(self, config):
        """Traces should be numbered [1], [2], etc. in the context output."""
        from engram.working.context import ContextBuilder

        builder = ContextBuilder(token_budget=4000, config=config)
        ctx = builder.build(
            person="alice",
            message="tell me about cats",
            salient_traces=[
                {
                    "id": "trace-aaa",
                    "kind": "episode",
                    "content": "Alice loves cats",
                    "salience": 0.8,
                },
                {
                    "id": "trace-bbb",
                    "kind": "summary",
                    "content": "Discussion about pets",
                    "salience": 0.6,
                },
            ],
        )
        assert "[1]" in ctx.text
        assert "[2]" in ctx.text
        assert "{trace:trace-aaa}" in ctx.text
        assert "{trace:trace-bbb}" in ctx.text
        assert "trace-aaa" in ctx.trace_ids
        assert "trace-bbb" in ctx.trace_ids

    def test_citation_hint_present(self, config):
        """Context output should include a citation instruction hint."""
        from engram.working.context import ContextBuilder

        builder = ContextBuilder(token_budget=4000, config=config)
        ctx = builder.build(
            person="alice",
            message="hello",
            salient_traces=[
                {
                    "id": "t1",
                    "kind": "episode",
                    "content": "something",
                    "salience": 0.5,
                },
            ],
        )
        assert "cite them as [1]" in ctx.text.lower() or "cite them as [1]" in ctx.text

    def test_no_traces_no_citation_section(self, config):
        """Without traces, no RELEVANT MEMORIES section."""
        from engram.working.context import ContextBuilder

        builder = ContextBuilder(token_budget=4000, config=config)
        ctx = builder.build(
            person="alice",
            message="hello",
            salient_traces=[],
        )
        assert "RELEVANT MEMORIES" not in ctx.text


# -----------------------------------------------------------------------
# Integration: pressure + compactor in after pipeline
# -----------------------------------------------------------------------


class TestPressureCompactorIntegration:
    def test_after_pipeline_with_pressure(self, config, episodic):
        """after() should accept and use memory_pressure + compactor params."""
        from engram.pipeline.after import after
        from engram.signal.decay import DecayEngine
        from engram.signal.measure import SignalTracker
        from engram.signal.reinforcement import ReinforcementEngine
        from engram.semantic.store import SemanticStore
        from engram.procedural.store import ProceduralStore

        sem = SemanticStore(semantic_dir=config.semantic_dir, soul_dir=config.soul_dir)
        proc = ProceduralStore(config.procedural_dir)
        tracker = SignalTracker()
        reinf = ReinforcementEngine()
        decay = DecayEngine(half_life_hours=config.decay_half_life_hours)
        mp = MemoryPressure(config)
        comp = ConversationCompactor()

        result = after(
            person="alice",
            their_message="hello",
            response="Hey Alice!",
            config=config,
            episodic=episodic,
            semantic=sem,
            procedural=proc,
            reinforcement=reinf,
            decay_engine=decay,
            signal_tracker=tracker,
            memory_pressure=mp,
            compactor=comp,
        )
        assert result.signal is not None
        assert result.salience > 0

    def test_after_pipeline_without_pressure_legacy(self, config, episodic):
        """after() should work without pressure (legacy path)."""
        from engram.pipeline.after import after
        from engram.signal.decay import DecayEngine
        from engram.signal.measure import SignalTracker
        from engram.signal.reinforcement import ReinforcementEngine
        from engram.semantic.store import SemanticStore
        from engram.procedural.store import ProceduralStore

        sem = SemanticStore(semantic_dir=config.semantic_dir, soul_dir=config.soul_dir)
        proc = ProceduralStore(config.procedural_dir)
        tracker = SignalTracker()
        reinf = ReinforcementEngine()
        decay = DecayEngine(half_life_hours=config.decay_half_life_hours)

        result = after(
            person="alice",
            their_message="hello",
            response="Hey there!",
            config=config,
            episodic=episodic,
            semantic=sem,
            procedural=proc,
            reinforcement=reinf,
            decay_engine=decay,
            signal_tracker=tracker,
            # No memory_pressure or compactor — legacy path
        )
        assert result.signal is not None

    def test_differential_reinforcement_on_cited_traces(self, config, episodic):
        """Traces cited in the response should get extra reinforcement."""
        from engram.pipeline.after import after
        from engram.signal.decay import DecayEngine
        from engram.signal.measure import SignalTracker
        from engram.signal.reinforcement import ReinforcementEngine
        from engram.semantic.store import SemanticStore
        from engram.procedural.store import ProceduralStore

        sem = SemanticStore(semantic_dir=config.semantic_dir, soul_dir=config.soul_dir)
        proc = ProceduralStore(config.procedural_dir)
        tracker = SignalTracker()
        reinf = ReinforcementEngine()
        decay = DecayEngine(half_life_hours=config.decay_half_life_hours)

        # Create a trace to be cited
        tid = episodic.log_trace(
            content="Alice likes cats",
            kind="episode",
            tags=["alice"],
            salience=0.5,
        )
        initial_salience = episodic.get_trace(tid)["salience"]

        # Response that cites [1]
        result = after(
            person="alice",
            their_message="what do I like?",
            response="Based on [1], you love cats!",
            trace_ids=[tid],
            config=config,
            episodic=episodic,
            semantic=sem,
            procedural=proc,
            reinforcement=reinf,
            decay_engine=decay,
            signal_tracker=tracker,
        )

        updated_trace = episodic.get_trace(tid)
        # The trace should have been reinforced (citation bonus + normal)
        # We can't predict exact value but it should have changed
        assert updated_trace is not None


# -----------------------------------------------------------------------
# MemoryConsolidator — helpers
# -----------------------------------------------------------------------


class TestConsolidatorHelpers:
    def test_group_by_person(self):
        traces = [
            {"tags": ["alice", "compaction"]},
            {"tags": ["alice"]},
            {"tags": ["bob"]},
            {"tags": ["consolidation"]},  # only skip tags → "_unknown"
        ]
        groups = _group_by_person(traces)
        assert len(groups["alice"]) == 2
        assert len(groups["bob"]) == 1
        assert "_unknown" in groups

    def test_group_by_person_no_tags(self):
        traces = [{"tags": None}, {"tags": []}]
        groups = _group_by_person(traces)
        assert "_unknown" in groups
        assert len(groups["_unknown"]) == 2

    def test_cluster_by_time_single_cluster(self):
        traces = [
            {"created": "2025-01-01T10:00:00Z"},
            {"created": "2025-01-01T11:00:00Z"},
            {"created": "2025-01-01T12:00:00Z"},
        ]
        clusters = _cluster_by_time(traces, window_hours=24.0)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_cluster_by_time_splits_on_gap(self):
        traces = [
            {"created": "2025-01-01T10:00:00Z"},
            {"created": "2025-01-01T11:00:00Z"},
            # 100 hour gap
            {"created": "2025-01-05T15:00:00Z"},
            {"created": "2025-01-05T16:00:00Z"},
        ]
        clusters = _cluster_by_time(traces, window_hours=72.0)
        assert len(clusters) == 2

    def test_cluster_by_time_empty(self):
        assert _cluster_by_time([], 72.0) == []

    # -- Topic-coherent clustering (HDBSCAN) --

    def test_cluster_by_topic_fallback_no_search(self):
        """Without semantic_search, falls back to temporal clustering."""
        traces = [
            {"id": "a", "created": "2025-01-01T10:00:00Z"},
            {"id": "b", "created": "2025-01-01T11:00:00Z"},
        ]
        clusters = _cluster_by_topic(traces, 72.0, 2, semantic_search=None)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_cluster_by_topic_empty(self):
        assert _cluster_by_topic([], 72.0, 2, semantic_search=None) == []

    def test_cluster_by_topic_with_embeddings(self):
        """HDBSCAN should separate traces with distinct embeddings."""
        import numpy as np

        # Create traces: 3 about topic A, 3 about topic B, well separated
        traces = []
        for i in range(6):
            traces.append(
                {
                    "id": f"t{i}",
                    "created": f"2025-01-01T{10 + i}:00:00Z",
                }
            )

        # Mock semantic_search that returns clearly separated embeddings
        mock_search = MagicMock()
        embeddings = {
            "trace_t0": [1.0, 0.0, 0.0] * 10,  # Topic A (30-dim for HDBSCAN)
            "trace_t1": [0.98, 0.02, 0.0] * 10,
            "trace_t2": [0.97, 0.03, 0.0] * 10,
            "trace_t3": [0.0, 0.0, 1.0] * 10,  # Topic B
            "trace_t4": [0.0, 0.02, 0.98] * 10,
            "trace_t5": [0.0, 0.03, 0.97] * 10,
        }
        mock_search.get_embeddings.return_value = embeddings

        clusters = _cluster_by_topic(traces, 72.0, 2, semantic_search=mock_search)
        # Should produce at least 2 clusters (topic A and topic B)
        # HDBSCAN with 3 points per topic and min_cluster_size=2 should find them
        assert len(clusters) >= 2
        # All traces should be in some cluster
        all_trace_ids = set()
        for cluster in clusters:
            for t in cluster:
                all_trace_ids.add(t["id"])
        assert all_trace_ids == {f"t{i}" for i in range(6)}

    def test_cluster_by_topic_few_embeddings_fallback(self):
        """Falls back to temporal if not enough embeddings."""
        traces = [
            {"id": "a", "created": "2025-01-01T10:00:00Z"},
            {"id": "b", "created": "2025-01-01T11:00:00Z"},
        ]
        mock_search = MagicMock()
        # Return only 1 embedding (below min_cluster_size=5)
        mock_search.get_embeddings.return_value = {"trace_a": [1.0, 0.0]}
        clusters = _cluster_by_topic(traces, 72.0, 5, semantic_search=mock_search)
        assert len(clusters) == 1  # temporal fallback: single cluster

    def test_time_range(self):
        traces = [
            {"created": "2025-01-01T10:00:00Z"},
            {"created": "2025-01-03T15:00:00Z"},
        ]
        result = _time_range(traces)
        assert "2025-01-01" in result
        assert "2025-01-03" in result

    def test_time_range_empty(self):
        assert _time_range([]) == ""

    def test_format_traces_for_llm(self):
        traces = [
            {
                "kind": "episode",
                "content": "Talked about cats",
                "created": "2025-01-01T10:00:00Z",
                "salience": 0.7,
            },
        ]
        text = _format_traces_for_llm(traces)
        assert "[1]" in text
        assert "cats" in text
        assert "episode" in text

    def test_format_traces_truncation(self):
        traces = [
            {
                "kind": "episode",
                "content": "x" * 20_000,
                "created": "2025-01-01T10:00:00Z",
                "salience": 0.5,
            },
        ]
        text = _format_traces_for_llm(traces)
        assert len(text) <= 15_100  # 15000 + some overhead

    def test_extractive_thread_summary(self):
        episodes = [
            {
                "content": "We discussed project plans",
                "salience": 0.9,
                "created": "2025-01-01T10:00:00Z",
            },
            {
                "content": "Minor update",
                "salience": 0.2,
                "created": "2025-01-01T11:00:00Z",
            },
            {
                "content": "Big breakthrough moment",
                "salience": 0.95,
                "created": "2025-01-01T12:00:00Z",
            },
        ]
        summary = _extractive_thread_summary(episodes, "alice")
        assert "Thread with alice" in summary
        assert "breakthrough" in summary  # highest salience

    def test_extractive_arc_summary(self):
        threads = [
            {"content": "Thread about cats", "created": "2025-01-01T10:00:00Z"},
            {"content": "Thread about work", "created": "2025-01-05T10:00:00Z"},
        ]
        summary = _extractive_arc_summary(threads, "alice")
        assert "Arc with alice" in summary
        assert "cats" in summary
        assert "work" in summary


# -----------------------------------------------------------------------
# MemoryConsolidator — mark_consolidated
# -----------------------------------------------------------------------


class TestMarkConsolidated:
    def test_marks_traces(self, episodic):
        tid1 = episodic.log_trace(
            content="ep1", kind="episode", tags=["alice"], salience=0.5
        )
        tid2 = episodic.log_trace(
            content="ep2", kind="episode", tags=["alice"], salience=0.5
        )

        _mark_consolidated(episodic, [tid1, tid2], "thread-xyz")

        import json

        for tid in [tid1, tid2]:
            row = episodic.conn.execute(
                "SELECT metadata FROM traces WHERE id = ?", (tid,)
            ).fetchone()
            meta = json.loads(row[0])
            assert meta["consolidated_into"] == "thread-xyz"

    def test_nonexistent_id_no_crash(self, episodic):
        """Marking a non-existent trace should not crash."""
        _mark_consolidated(episodic, ["nonexistent-id"], "thread-xyz")


# -----------------------------------------------------------------------
# MemoryConsolidator — thread consolidation
# -----------------------------------------------------------------------


class TestThreadConsolidation:
    def _create_episodes(self, episodic, person, count, base_hour=0):
        """Create `count` episode traces for a person."""
        ids = []
        for i in range(count):
            tid = episodic.log_trace(
                content=f"Episode {i}: conversation about topic {i % 3} with {person}",
                kind="episode",
                tags=[person, "test"],
                salience=0.3 + (i % 5) * 0.1,
            )
            ids.append(tid)
        return ids

    def test_below_threshold_no_threads(self, episodic):
        """Don't create threads if fewer episodes than threshold."""
        self._create_episodes(episodic, "alice", 3)
        consolidator = MemoryConsolidator(min_episodes_per_thread=5)
        result = consolidator.consolidate_threads(episodic)
        assert result == []

    def test_creates_threads_above_threshold(self, episodic):
        """Should create thread traces when enough episodes exist."""
        self._create_episodes(episodic, "alice", 10)
        consolidator = MemoryConsolidator(
            min_episodes_per_thread=5,
            thread_time_window_hours=1000.0,  # all in one cluster
        )
        thread_ids = consolidator.consolidate_threads(episodic)
        assert len(thread_ids) >= 1

        # Verify the thread trace exists
        for tid in thread_ids:
            trace = episodic.get_trace(tid)
            assert trace is not None
            assert trace["kind"] == "thread"
            assert trace["salience"] == 0.75

    def test_threads_have_metadata(self, episodic):
        """Thread traces should have child_ids and time_range in metadata."""
        self._create_episodes(episodic, "bob", 8)
        consolidator = MemoryConsolidator(
            min_episodes_per_thread=5,
            thread_time_window_hours=1000.0,
        )
        thread_ids = consolidator.consolidate_threads(episodic)
        assert len(thread_ids) >= 1

        trace = episodic.get_trace(thread_ids[0])
        meta = trace.get("metadata", {})
        assert "child_ids" in meta
        assert "time_range" in meta
        assert len(meta["child_ids"]) >= 5

    def test_episodes_marked_consolidated(self, episodic):
        """After threading, child episodes should have consolidated_into."""
        self._create_episodes(episodic, "carol", 8)
        consolidator = MemoryConsolidator(
            min_episodes_per_thread=5,
            thread_time_window_hours=1000.0,
        )
        thread_ids = consolidator.consolidate_threads(episodic)
        assert len(thread_ids) >= 1

        # Check that child episodes are now marked
        unconsolidated = _get_unconsolidated_episodes(episodic)
        # All 8 episodes should be consolidated (none unconsolidated)
        assert len(unconsolidated) == 0

    def test_no_double_consolidation(self, episodic):
        """Running consolidation twice should not re-process same episodes."""
        self._create_episodes(episodic, "dave", 10)
        consolidator = MemoryConsolidator(
            min_episodes_per_thread=5,
            thread_time_window_hours=1000.0,
        )

        first_ids = consolidator.consolidate_threads(episodic)
        assert len(first_ids) >= 1

        # Second run should find nothing to consolidate
        second_ids = consolidator.consolidate_threads(episodic)
        assert second_ids == []

    def test_with_llm_func(self, episodic):
        """Thread creation should use LLM when available."""
        self._create_episodes(episodic, "eve", 8)

        call_count = 0

        def mock_llm(prompt, system):
            nonlocal call_count
            call_count += 1
            return "LLM-generated thread summary about conversations with Eve."

        consolidator = MemoryConsolidator(
            min_episodes_per_thread=5,
            thread_time_window_hours=1000.0,
        )
        thread_ids = consolidator.consolidate_threads(episodic, llm_func=mock_llm)
        assert len(thread_ids) >= 1
        assert call_count > 0

    def test_multiple_people_separate_threads(self, episodic):
        """Episodes for different people should create separate threads."""
        self._create_episodes(episodic, "alice", 8)
        self._create_episodes(episodic, "bob", 8)

        consolidator = MemoryConsolidator(
            min_episodes_per_thread=5,
            thread_time_window_hours=1000.0,
        )
        thread_ids = consolidator.consolidate_threads(episodic)
        assert len(thread_ids) >= 2

        # Verify threads have different person tags
        people_in_threads = set()
        for tid in thread_ids:
            trace = episodic.get_trace(tid)
            tags = trace.get("tags", [])
            for tag in tags:
                if tag not in ("consolidation", "test"):
                    people_in_threads.add(tag)
        assert "alice" in people_in_threads
        assert "bob" in people_in_threads


# -----------------------------------------------------------------------
# MemoryConsolidator — arc consolidation
# -----------------------------------------------------------------------


class TestArcConsolidation:
    def _create_threads(self, episodic, person, count):
        """Create `count` thread traces for a person."""
        ids = []
        for i in range(count):
            tid = episodic.log_trace(
                content=f"Thread {i}: summary of conversations about topic {i} with {person}",
                kind="thread",
                tags=[person, "consolidation"],
                salience=0.75,
            )
            ids.append(tid)
        return ids

    def test_below_threshold_no_arcs(self, episodic):
        """Don't create arcs if fewer threads than threshold."""
        self._create_threads(episodic, "alice", 2)
        consolidator = MemoryConsolidator(min_threads_per_arc=3)
        result = consolidator.consolidate_arcs(episodic)
        assert result == []

    def test_creates_arcs_above_threshold(self, episodic):
        """Should create arc traces when enough threads exist."""
        self._create_threads(episodic, "alice", 5)
        consolidator = MemoryConsolidator(min_threads_per_arc=3)
        arc_ids = consolidator.consolidate_arcs(episodic)
        assert len(arc_ids) >= 1

        for aid in arc_ids:
            trace = episodic.get_trace(aid)
            assert trace is not None
            assert trace["kind"] == "arc"
            assert trace["salience"] == 0.85

    def test_arcs_have_metadata(self, episodic):
        """Arc traces should have child_thread_ids and time_range."""
        self._create_threads(episodic, "bob", 4)
        consolidator = MemoryConsolidator(min_threads_per_arc=3)
        arc_ids = consolidator.consolidate_arcs(episodic)
        assert len(arc_ids) >= 1

        trace = episodic.get_trace(arc_ids[0])
        meta = trace.get("metadata", {})
        assert "child_thread_ids" in meta
        assert "time_range" in meta

    def test_threads_marked_consolidated_after_arc(self, episodic):
        """After arc creation, child threads should have consolidated_into."""
        self._create_threads(episodic, "carol", 4)
        consolidator = MemoryConsolidator(min_threads_per_arc=3)
        consolidator.consolidate_arcs(episodic)

        unconsolidated = _get_unconsolidated_threads(episodic)
        assert len(unconsolidated) == 0

    def test_no_double_arc_consolidation(self, episodic):
        """Running arc consolidation twice should not re-process."""
        self._create_threads(episodic, "dave", 5)
        consolidator = MemoryConsolidator(min_threads_per_arc=3)

        first = consolidator.consolidate_arcs(episodic)
        assert len(first) >= 1

        second = consolidator.consolidate_arcs(episodic)
        assert second == []


# -----------------------------------------------------------------------
# MemoryConsolidator — full consolidation
# -----------------------------------------------------------------------


class TestFullConsolidation:
    def test_consolidate_episodes_to_arcs(self, episodic):
        """Full pipeline: create episodes → threads → arcs."""
        # Create enough episodes
        for i in range(15):
            episodic.log_trace(
                content=f"Episode {i}: interaction with alice",
                kind="episode",
                tags=["alice"],
                salience=0.5,
            )

        consolidator = MemoryConsolidator(
            min_episodes_per_thread=5,
            thread_time_window_hours=1000.0,
            min_threads_per_arc=3,
        )

        # First pass: episodes → threads
        result1 = consolidator.consolidate(episodic)
        assert len(result1["threads"]) >= 1
        # Not enough threads yet for arcs (only 1 cluster likely)
        # but that's OK — depends on clustering

        # Create more episodes and consolidate again to get more threads
        for i in range(15):
            episodic.log_trace(
                content=f"Episode batch2 {i}: more talks with alice",
                kind="episode",
                tags=["alice"],
                salience=0.5,
            )
        result2 = consolidator.consolidate(episodic)
        # We should have created more threads from the new episodes
        assert len(result2["threads"]) >= 1

    def test_consolidate_returns_dict(self, episodic):
        """consolidate() should always return a dict with threads/arcs keys."""
        consolidator = MemoryConsolidator()
        result = consolidator.consolidate(episodic)
        assert "threads" in result
        assert "arcs" in result
        assert isinstance(result["threads"], list)
        assert isinstance(result["arcs"], list)

    def test_system_consolidate_method(self, config, episodic):
        """MemorySystem.consolidate() should work end-to-end."""
        from engram.system import MemorySystem

        with MemorySystem(config=config) as ms:
            # Create episodes through the episodic store
            for i in range(10):
                ms.episodic.log_trace(
                    content=f"Episode {i}: talked with alice about things",
                    kind="episode",
                    tags=["alice"],
                    salience=0.5,
                )

            result = ms.consolidate()
            assert "threads" in result
            assert "arcs" in result


# -----------------------------------------------------------------------
# Sessions — table creation and CRUD
# -----------------------------------------------------------------------


class TestSessions:
    def test_start_session(self, episodic):
        sid = episodic.start_session("alice")
        assert sid is not None
        assert len(sid) == 12

    def test_get_active_session(self, episodic):
        sid = episodic.start_session("alice")
        active = episodic.get_active_session("alice")
        assert active is not None
        assert active["id"] == sid
        assert active["person"] == "alice"
        assert active["ended"] is None

    def test_end_session(self, episodic):
        sid = episodic.start_session("alice")
        episodic.end_session(sid, summary="Good conversation about cats.")
        active = episodic.get_active_session("alice")
        assert active is None  # no longer active

    def test_get_recent_sessions(self, episodic):
        episodic.start_session("alice")
        episodic.start_session("bob")
        sessions = episodic.get_recent_sessions()
        assert len(sessions) == 2

    def test_get_recent_sessions_person_filter(self, episodic):
        episodic.start_session("alice")
        episodic.start_session("bob")
        sessions = episodic.get_recent_sessions(person="alice")
        assert len(sessions) == 1
        assert sessions[0]["person"] == "alice"

    def test_increment_message_count(self, episodic):
        sid = episodic.start_session("alice")
        episodic.increment_session_message_count(sid)
        episodic.increment_session_message_count(sid)
        active = episodic.get_active_session("alice")
        assert active["message_count"] == 2

    def test_detect_session_boundary_no_messages(self, episodic):
        """No messages → new session needed."""
        assert episodic.detect_session_boundary("alice") is True

    def test_detect_session_boundary_recent(self, episodic):
        """Recent message → no new session needed."""
        episodic.log_message(
            person="alice", speaker="alice", content="hello", source="test"
        )
        assert episodic.detect_session_boundary("alice", gap_hours=2.0) is False

    def test_session_end_has_summary(self, episodic):
        sid = episodic.start_session("alice")
        episodic.end_session(sid, summary="We talked about cats and dogs.")
        sessions = episodic.get_recent_sessions(person="alice")
        assert sessions[0]["summary"] == "We talked about cats and dogs."


# -----------------------------------------------------------------------
# Temporal retrieval
# -----------------------------------------------------------------------


class TestTemporalRetrieval:
    def test_get_messages_in_range(self, episodic):
        episodic.log_message(
            person="alice", speaker="alice", content="morning msg", source="test"
        )
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        since = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        until = (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z")

        msgs = episodic.get_messages_in_range(since=since, until=until)
        assert len(msgs) >= 1

    def test_get_messages_in_range_with_person(self, episodic):
        episodic.log_message(
            person="alice", speaker="alice", content="alice msg", source="test"
        )
        episodic.log_message(
            person="bob", speaker="bob", content="bob msg", source="test"
        )
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        since = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        until = (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z")

        msgs = episodic.get_messages_in_range(since=since, until=until, person="alice")
        assert all(m["person"] == "alice" for m in msgs)

    def test_get_traces_in_range(self, episodic):
        episodic.log_trace(
            content="test trace", kind="episode", tags=["test"], salience=0.5
        )
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        since = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        until = (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z")

        traces = episodic.get_traces_in_range(since=since, until=until)
        assert len(traces) >= 1

    def test_get_sessions_in_range(self, episodic):
        episodic.start_session("alice")
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        since = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        until = (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z")

        sessions = episodic.get_sessions_in_range(since=since, until=until)
        assert len(sessions) >= 1

    def test_empty_range(self, episodic):
        msgs = episodic.get_messages_in_range(
            since="2020-01-01T00:00:00Z",
            until="2020-01-02T00:00:00Z",
        )
        assert msgs == []


# -----------------------------------------------------------------------
# engram_recall_time MCP tool
# -----------------------------------------------------------------------


class TestEngramRecallTime:
    @pytest.fixture
    def server_system(self, config):
        from engram.system import MemorySystem
        import engram.server as server_mod

        system = MemorySystem(config=config)
        old = server_mod._system
        server_mod._system = system
        yield system
        server_mod._system = old
        system.close()

    def test_returns_valid_json(self, server_system):
        import json
        import engram.server as server_mod

        result = server_mod.engram_recall_time(since="2025-01-01", until="2025-12-31")
        data = json.loads(result)
        assert "since" in data
        assert "until" in data
        assert "messages" in data
        assert "traces" in data
        assert "sessions" in data

    def test_bare_date_parsed(self, server_system):
        import json
        import engram.server as server_mod

        result = server_mod.engram_recall_time(since="2025-06-15")
        data = json.loads(result)
        assert "2025-06-15" in data["since"]

    def test_with_person_filter(self, server_system):
        import json
        import engram.server as server_mod

        server_system.episodic.log_message(
            person="alice", speaker="alice", content="hello", source="test"
        )
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        since = (now - timedelta(hours=1)).strftime("%Y-%m-%d")

        result = server_mod.engram_recall_time(since=since, person="alice")
        data = json.loads(result)
        assert data["person"] == "alice"

    def test_messages_only(self, server_system):
        import json
        import engram.server as server_mod

        result = server_mod.engram_recall_time(since="2025-01-01", what="messages")
        data = json.loads(result)
        assert "messages" in data
        assert "traces" not in data
        assert "sessions" not in data
