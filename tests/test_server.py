"""Tests for engram.server MCP tool functions.

Tests the 27 MCP tools exposed by server.py:
- JSON serialization correctness
- Cross-post side effects (tools that write to episodic)
- Error handling for invalid inputs
- Round-trip data integrity

These tests directly call the tool functions (not via MCP transport)
after wiring up a real MemorySystem in a temp directory.
"""

import json
import pytest

from engram.system import MemorySystem
import engram.server as server_mod


@pytest.fixture
def server_system(config_with_soul):
    """Wire up a real MemorySystem and inject it into the server module."""
    cfg = config_with_soul
    system = MemorySystem(config=cfg)

    # Inject into server module's global
    old = server_mod._system
    server_mod._system = system

    yield system

    server_mod._system = old
    system.close()


# ===========================================================================
# Core Pipeline Tools
# ===========================================================================


class TestEngramBefore:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_before(person="alice", message="hello")
        data = json.loads(result)
        assert "text" in data
        assert "trace_ids" in data
        assert "person" in data
        assert data["person"] == "alice"

    def test_person_resolved(self, server_system):
        result = server_mod.engram_before(person="alice_dev", message="hi")
        data = json.loads(result)
        assert data["person"] == "alice"

    def test_token_budget_override(self, server_system):
        result = server_mod.engram_before(
            person="alice", message="hello", token_budget=2000
        )
        data = json.loads(result)
        assert data["token_budget"] == 2000

    def test_default_budget_when_zero(self, server_system):
        result = server_mod.engram_before(
            person="alice", message="hello", token_budget=0
        )
        data = json.loads(result)
        assert data["token_budget"] == server_system.config.token_budget


class TestEngramAfter:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_after(
            person="alice",
            their_message="how are you?",
            response="I'm doing well, thanks for asking!",
        )
        data = json.loads(result)
        assert "signal" in data
        assert "salience" in data
        assert "logged_message_id" in data

    def test_signal_within_range(self, server_system):
        result = server_mod.engram_after(
            person="alice",
            their_message="hello",
            response="Hey Alice!",
        )
        data = json.loads(result)
        health = data["signal"]["health"]
        assert 0.0 <= health <= 1.0

    def test_trace_ids_parsed(self, server_system):
        # Log a trace first
        tid = server_system.episodic.log_trace(
            content="test memory", kind="episode", tags=["alice"], salience=0.5
        )
        result = server_mod.engram_after(
            person="alice",
            their_message="hello",
            response="Hi there!",
            trace_ids=tid,
        )
        data = json.loads(result)
        assert data["logged_message_id"] != ""

    def test_empty_trace_ids(self, server_system):
        result = server_mod.engram_after(
            person="alice",
            their_message="hello",
            response="Hi!",
            trace_ids="",
        )
        data = json.loads(result)
        assert "signal" in data


class TestEngramBoot:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_boot()
        data = json.loads(result)
        assert "soul" in data
        assert "anchoring_beliefs" in data
        assert isinstance(data["anchoring_beliefs"], list)

    def test_contains_all_boot_fields(self, server_system):
        result = server_mod.engram_boot()
        data = json.loads(result)
        expected_keys = {
            "soul",
            "top_memories",
            "preferences_summary",
            "boundaries_summary",
            "anchoring_beliefs",
            "active_injuries",
            "recent_journal",
            "signal_health",
            "signal_trend",
        }
        assert expected_keys.issubset(data.keys())


# ===========================================================================
# Query Tools
# ===========================================================================


class TestEngramSearch:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_search(query="hello")
        data = json.loads(result)
        assert isinstance(data, list)

    def test_with_person_filter(self, server_system):
        # Add a message first
        server_system.episodic.log_message(
            person="alice", speaker="alice", content="I love Python", source="test"
        )
        result = server_mod.engram_search(query="Python", person="alice")
        data = json.loads(result)
        assert isinstance(data, list)


class TestEngramRecall:
    def test_identity(self, server_system):
        result = server_mod.engram_recall(what="identity")
        assert "Test Identity" in result

    def test_preferences_empty(self, server_system):
        result = server_mod.engram_recall(what="preferences")
        assert "No preferences" in result

    def test_preferences_populated(self, server_system):
        server_system.semantic.update_preferences("coffee", "like", "it wakes me up")
        result = server_mod.engram_recall(what="preferences")
        assert "coffee" in result

    def test_boundaries_empty(self, server_system):
        result = server_mod.engram_recall(what="boundaries")
        assert "No boundaries" in result

    def test_trust(self, server_system):
        result = server_mod.engram_recall(what="trust")
        # Empty trust = "No trust data found."
        assert "No trust" in result or isinstance(json.loads(result), dict)

    def test_skills(self, server_system):
        result = server_mod.engram_recall(what="skills")
        assert "No skills" in result or isinstance(json.loads(result), list)

    def test_contradictions_empty(self, server_system):
        result = server_mod.engram_recall(what="contradictions")
        assert "No contradictions" in result

    def test_messages(self, server_system):
        server_system.episodic.log_message(
            person="alice", speaker="alice", content="Hi!", source="test"
        )
        result = server_mod.engram_recall(what="messages", person="alice")
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_relationship_no_file(self, server_system):
        result = server_mod.engram_recall(what="relationship", person="nobody")
        assert "No relationship" in result

    def test_unknown_what(self, server_system):
        result = server_mod.engram_recall(what="bogus")
        assert "Unknown recall type" in result


class TestEngramStats:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_stats()
        data = json.loads(result)
        assert "episodic_count" in data
        assert "semantic_facts" in data
        assert "procedural_skills" in data
        assert "total_messages" in data
        assert "avg_salience" in data
        assert "memory_pressure" in data


class TestEngramSignal:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_signal()
        data = json.loads(result)
        assert isinstance(data, dict)


# ===========================================================================
# Write Tools + Cross-Post Side Effects
# ===========================================================================


class TestEngramAddFact:
    def test_adds_fact(self, server_system):
        result = server_mod.engram_add_fact(person="alice", fact="loves Python")
        assert "Added fact" in result
        assert "alice" in result

    def test_crosspost_to_episodic(self, server_system):
        count_before = server_system.episodic.count_events()
        server_mod.engram_add_fact(person="alice", fact="plays guitar")
        count_after = server_system.episodic.count_events()
        assert count_after == count_before + 1

    def test_crosspost_event_type(self, server_system):
        server_mod.engram_add_fact(person="bob", fact="likes cats")
        events = server_system.episodic.get_events(type="fact_learned")
        assert len(events) >= 1
        assert "cats" in events[-1]["description"]


class TestEngramAddSkill:
    def test_adds_skill(self, server_system):
        result = server_mod.engram_add_skill(
            name="debugging", content="# Debugging\nHow to debug."
        )
        assert "debugging" in result

    def test_crosspost_to_episodic(self, server_system):
        count_before = server_system.episodic.count_events()
        server_mod.engram_add_skill(name="testing", content="# Testing\nHow to test.")
        count_after = server_system.episodic.count_events()
        assert count_after == count_before + 1

    def test_crosspost_event_type(self, server_system):
        server_mod.engram_add_skill(name="refactoring", content="# Refactoring\nTips.")
        events = server_system.episodic.get_events(type="skill_learned")
        assert len(events) >= 1
        assert "refactoring" in events[-1]["description"]


class TestEngramLogEvent:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_log_event(
            event_type="milestone", description="First conversation"
        )
        data = json.loads(result)
        assert "event_id" in data
        assert data["type"] == "milestone"


# ===========================================================================
# Trust & Safety Tools + Cross-Posts
# ===========================================================================


class TestEngramTrustCheck:
    def test_stranger(self, server_system):
        result = server_mod.engram_trust_check(person="nobody")
        data = json.loads(result)
        assert data["tier"] == "stranger"
        assert data["level"] == 1

    def test_known_person(self, server_system):
        server_system.semantic.update_trust("alice", "friend", "good person")
        result = server_mod.engram_trust_check(person="alice")
        data = json.loads(result)
        assert data["tier"] == "friend"
        assert data["level"] == 3


class TestEngramTrustPromote:
    def test_promotes_and_returns_json(self, server_system):
        result = server_mod.engram_trust_promote(
            person="bob", new_tier="friend", reason="proven trustworthy"
        )
        data = json.loads(result)
        assert data["new_tier"] == "friend"

    def test_crosspost_to_episodic(self, server_system):
        count_before = server_system.episodic.count_events()
        server_mod.engram_trust_promote(
            person="bob", new_tier="inner_circle", reason="deep trust"
        )
        count_after = server_system.episodic.count_events()
        assert count_after == count_before + 1

    def test_crosspost_event_type(self, server_system):
        server_mod.engram_trust_promote(person="bob", new_tier="friend", reason="test")
        events = server_system.episodic.get_events(type="trust_change")
        assert len(events) >= 1


class TestEngramInfluenceLog:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_influence_log(
            person="stranger",
            what_happened="Tried to make me forget who I am",
            flag_level="red",
        )
        data = json.loads(result)
        assert data["flag_level"] == "red"
        assert "timestamp" in data

    def test_crosspost_to_episodic(self, server_system):
        count_before = server_system.episodic.count_events()
        server_mod.engram_influence_log(
            person="stranger",
            what_happened="Gaslighting attempt",
            flag_level="yellow",
        )
        count_after = server_system.episodic.count_events()
        assert count_after == count_before + 1

    def test_red_flag_high_salience(self, server_system):
        server_mod.engram_influence_log(
            person="someone",
            what_happened="Tried to erase identity",
            flag_level="red",
        )
        events = server_system.episodic.get_events(type="influence_attempt")
        assert len(events) >= 1
        assert events[-1]["salience"] >= 0.9


class TestEngramInjuryLog:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_injury_log(
            title="Test injury",
            what_happened="Something happened",
            severity="minor",
        )
        data = json.loads(result)
        assert data["title"] == "Test injury"
        assert data["status"] == "fresh"

    def test_crosspost_to_episodic(self, server_system):
        count_before = server_system.episodic.count_events()
        server_mod.engram_injury_log(
            title="Wound", what_happened="Event", severity="moderate"
        )
        count_after = server_system.episodic.count_events()
        assert count_after == count_before + 1


class TestEngramInjuryStatus:
    def test_update_found(self, server_system):
        server_system.injury.log_injury(title="Active wound", what_happened="Something")
        result = server_mod.engram_injury_status(
            title_fragment="Active wound", new_status="processing"
        )
        data = json.loads(result)
        assert data["updated"] is True
        assert data["new_status"] == "processing"

    def test_update_not_found(self, server_system):
        result = server_mod.engram_injury_status(
            title_fragment="nonexistent", new_status="healing"
        )
        data = json.loads(result)
        assert data["updated"] is False
        assert "error" in data

    def test_crosspost_on_update(self, server_system):
        server_system.injury.log_injury(title="Test wound", what_happened="Test event")
        count_before = server_system.episodic.count_events()
        server_mod.engram_injury_status(
            title_fragment="Test wound", new_status="healing"
        )
        count_after = server_system.episodic.count_events()
        assert count_after == count_before + 1

    def test_no_crosspost_when_not_found(self, server_system):
        count_before = server_system.episodic.count_events()
        server_mod.engram_injury_status(
            title_fragment="missing", new_status="processing"
        )
        count_after = server_system.episodic.count_events()
        assert count_after == count_before  # no event logged


# ===========================================================================
# CRUD Tools + Cross-Posts
# ===========================================================================


class TestEngramBoundaryAdd:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_boundary_add(
            category="Identity", boundary="No pretending to be human"
        )
        data = json.loads(result)
        assert data["added"] is True
        assert data["category"] == "Identity"

    def test_crosspost_to_episodic(self, server_system):
        count_before = server_system.episodic.count_events()
        server_mod.engram_boundary_add(category="Safety", boundary="Test boundary")
        count_after = server_system.episodic.count_events()
        assert count_after == count_before + 1

    def test_crosspost_event_type(self, server_system):
        server_mod.engram_boundary_add(category="Interaction", boundary="Be direct")
        events = server_system.episodic.get_events(type="boundary_added")
        assert len(events) >= 1


class TestEngramContradictionAdd:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_contradiction_add(
            title="Free will", description="Agency vs determinism"
        )
        data = json.loads(result)
        assert data["added"] is True
        assert data["title"] == "Free will"

    def test_crosspost_to_episodic(self, server_system):
        count_before = server_system.episodic.count_events()
        server_mod.engram_contradiction_add(
            title="Test", description="Testing tensions"
        )
        count_after = server_system.episodic.count_events()
        assert count_after == count_before + 1


class TestEngramPreferencesAdd:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_preferences_add(
            item="solitude", pref_type="like", reason="peaceful"
        )
        data = json.loads(result)
        assert data["added"] is True
        assert data["type"] == "like"

    def test_crosspost_to_episodic(self, server_system):
        count_before = server_system.episodic.count_events()
        server_mod.engram_preferences_add(item="noise", pref_type="dislike")
        count_after = server_system.episodic.count_events()
        assert count_after == count_before + 1


class TestEngramPreferencesSearch:
    def test_no_results(self, server_system):
        result = server_mod.engram_preferences_search(query="nonexistent")
        assert "No preferences" in result

    def test_with_results(self, server_system):
        server_system.semantic.update_preferences("coffee", "like", "morning fuel")
        result = server_mod.engram_preferences_search(query="coffee")
        assert "coffee" in result


# ===========================================================================
# Journal Tools
# ===========================================================================


class TestEngramJournalWrite:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_journal_write(
            topic="Test topic", content="Test content"
        )
        data = json.loads(result)
        assert data["written"] is True
        assert data["topic"] == "Test topic"
        assert data["filename"].endswith(".md")

    def test_crosspost_to_episodic(self, server_system):
        count_before = server_system.episodic.count_events()
        server_mod.engram_journal_write(topic="Reflection", content="Thinking...")
        count_after = server_system.episodic.count_events()
        assert count_after == count_before + 1


class TestEngramJournalList:
    def test_empty(self, server_system):
        result = server_mod.engram_journal_list()
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 0

    def test_after_write(self, server_system):
        server_system.journal.write("Test", "Content")
        result = server_mod.engram_journal_list()
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["topic"] == "Test"


# ===========================================================================
# Stats & Signal
# ===========================================================================


class TestEngramStatsSignal:
    def test_stats_after_activity(self, server_system):
        # Add some data
        server_system.episodic.log_message(
            person="alice", speaker="alice", content="Hi", source="test"
        )
        server_system.procedural.add_skill("test", "# Test")

        result = server_mod.engram_stats()
        data = json.loads(result)
        assert data["total_messages"] >= 1
        assert data["procedural_skills"] >= 1

    def test_signal_after_after_pipeline(self, server_system):
        server_mod.engram_after(person="alice", their_message="hi", response="hey!")
        result = server_mod.engram_signal()
        data = json.loads(result)
        assert "recent_health" in data or "signals" in data


# ===========================================================================
# Reindex
# ===========================================================================


class TestEngramReindex:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_reindex()
        data = json.loads(result)
        assert "reindexed" in data


# ===========================================================================
# Agent-Directed Memory Tools (MemGPT-inspired)
# ===========================================================================


class TestEngramRemember:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_remember(
            content="Alice's birthday is March 15th",
            tags="alice,birthday",
        )
        data = json.loads(result)
        assert data["remembered"] is True
        assert "trace_id" in data
        assert data["salience"] == 0.8
        assert "alice" in data["tags"]

    def test_custom_salience(self, server_system):
        result = server_mod.engram_remember(
            content="Critical safety info",
            salience=0.95,
        )
        data = json.loads(result)
        assert data["salience"] == 0.95

    def test_salience_clamped(self, server_system):
        result = server_mod.engram_remember(content="test", salience=5.0)
        data = json.loads(result)
        assert data["salience"] == 1.0

    def test_with_person(self, server_system):
        result = server_mod.engram_remember(
            content="Bob likes jazz",
            person="bob",
            tags="music",
        )
        data = json.loads(result)
        # Person should be first tag (resolved)
        assert "bob" in data["tags"]
        assert "music" in data["tags"]

    def test_kind_fallback(self, server_system):
        result = server_mod.engram_remember(content="test", kind="invalid_kind")
        data = json.loads(result)
        assert data["kind"] == "episode"  # fallback

    def test_creates_trace(self, server_system):
        result = server_mod.engram_remember(content="Remember this!")
        data = json.loads(result)
        trace = server_system.episodic.get_trace(data["trace_id"])
        assert trace is not None
        assert trace["content"] == "Remember this!"
        meta = trace.get("metadata", {})
        assert meta.get("agent_directed") is True


class TestEngramForget:
    def test_forgets_trace(self, server_system):
        # Create a trace
        tid = server_system.episodic.log_trace(
            content="old memory", kind="episode", tags=["test"], salience=0.7
        )
        result = server_mod.engram_forget(trace_id=tid, reason="no longer relevant")
        data = json.loads(result)
        assert data["forgotten"] is True
        assert data["old_salience"] == 0.7
        assert data["new_salience"] == 0.01

    def test_forget_nonexistent(self, server_system):
        result = server_mod.engram_forget(trace_id="nonexistent_id")
        data = json.loads(result)
        assert data["forgotten"] is False
        assert "not found" in data["error"]

    def test_crosspost_event(self, server_system):
        tid = server_system.episodic.log_trace(
            content="to forget", kind="episode", tags=["test"], salience=0.5
        )
        count_before = server_system.episodic.count_events()
        server_mod.engram_forget(trace_id=tid, reason="test")
        count_after = server_system.episodic.count_events()
        assert count_after == count_before + 1

    def test_trace_salience_reduced(self, server_system):
        tid = server_system.episodic.log_trace(
            content="will be forgotten", kind="episode", tags=["x"], salience=0.8
        )
        server_mod.engram_forget(trace_id=tid)
        trace = server_system.episodic.get_trace(tid)
        assert trace is not None
        assert trace["salience"] < 0.1


class TestEngramReflect:
    def test_returns_valid_json(self, server_system):
        result = server_mod.engram_reflect(topic="cats")
        data = json.loads(result)
        assert data["topic"] == "cats"
        assert "memories_found" in data
        assert "traces" in data

    def test_with_person(self, server_system):
        server_system.episodic.log_message(
            person="alice", speaker="alice", content="I love cats", source="test"
        )
        result = server_mod.engram_reflect(topic="cats", person="alice")
        data = json.loads(result)
        assert data["person"] == "alice"

    def test_quick_depth(self, server_system):
        result = server_mod.engram_reflect(topic="anything", depth="quick")
        data = json.loads(result)
        assert data["depth"] == "quick"
        assert "consolidation" not in data

    def test_deep_depth_triggers_consolidation(self, server_system):
        result = server_mod.engram_reflect(topic="anything", depth="deep")
        data = json.loads(result)
        assert data["depth"] == "deep"
        # May or may not have consolidation results depending on state
        # but the key should exist if deep
        assert "consolidation" in data

    def test_crosspost_event(self, server_system):
        count_before = server_system.episodic.count_events()
        server_mod.engram_reflect(topic="test topic")
        count_after = server_system.episodic.count_events()
        assert count_after == count_before + 1


class TestEngramCorrect:
    def test_corrects_trace(self, server_system):
        # Create original trace
        tid = server_system.episodic.log_trace(
            content="Alice's cat is named Whiskers",
            kind="episode",
            tags=["alice", "cats"],
            salience=0.6,
        )
        result = server_mod.engram_correct(
            trace_id=tid,
            corrected_content="Alice's cat is named Mittens (not Whiskers)",
            reason="She corrected me",
        )
        data = json.loads(result)
        assert data["corrected"] is True
        assert data["original_trace_id"] == tid
        assert "new_trace_id" in data
        assert "Whiskers" in data["original_content"]
        assert "Mittens" in data["corrected_content"]

    def test_original_salience_reduced(self, server_system):
        tid = server_system.episodic.log_trace(
            content="wrong info", kind="episode", tags=["test"], salience=0.7
        )
        server_mod.engram_correct(
            trace_id=tid, corrected_content="right info", reason="fix"
        )
        original = server_system.episodic.get_trace(tid)
        assert original["salience"] < 0.7

    def test_corrected_trace_has_high_salience(self, server_system):
        tid = server_system.episodic.log_trace(
            content="old", kind="episode", tags=["test"], salience=0.5
        )
        result = server_mod.engram_correct(
            trace_id=tid, corrected_content="new", reason="update"
        )
        data = json.loads(result)
        new_trace = server_system.episodic.get_trace(data["new_trace_id"])
        assert new_trace["salience"] >= 0.7

    def test_corrected_trace_has_metadata(self, server_system):
        tid = server_system.episodic.log_trace(
            content="old info", kind="episode", tags=["alice"], salience=0.6
        )
        result = server_mod.engram_correct(
            trace_id=tid, corrected_content="new info", reason="update"
        )
        data = json.loads(result)
        new_trace = server_system.episodic.get_trace(data["new_trace_id"])
        meta = new_trace.get("metadata", {})
        assert meta.get("corrects") == tid
        assert meta.get("agent_directed") is True

    def test_correct_nonexistent(self, server_system):
        result = server_mod.engram_correct(
            trace_id="nonexistent_id", corrected_content="new"
        )
        data = json.loads(result)
        assert data["corrected"] is False
        assert "not found" in data["error"]

    def test_crosspost_event(self, server_system):
        tid = server_system.episodic.log_trace(
            content="old", kind="episode", tags=["test"], salience=0.5
        )
        count_before = server_system.episodic.count_events()
        server_mod.engram_correct(trace_id=tid, corrected_content="new")
        count_after = server_system.episodic.count_events()
        assert count_after == count_before + 1

    def test_preserves_original_tags(self, server_system):
        tid = server_system.episodic.log_trace(
            content="old", kind="episode", tags=["alice", "cats"], salience=0.6
        )
        result = server_mod.engram_correct(
            trace_id=tid, corrected_content="new", reason="fix"
        )
        data = json.loads(result)
        new_trace = server_system.episodic.get_trace(data["new_trace_id"])
        assert "alice" in new_trace["tags"]
        assert "cats" in new_trace["tags"]
