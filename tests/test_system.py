"""Integration tests for engram.system.MemorySystem."""

import pytest
from engram.system import MemorySystem
from engram.core.config import Config
from engram.core.types import Context, AfterResult, MemoryStats


@pytest.fixture
def memory(config_with_soul):
    """Provide a fully initialized MemorySystem."""
    mem = MemorySystem(config=config_with_soul)
    yield mem
    mem.close()


class TestMemorySystem:
    def test_init(self, memory):
        assert memory.config is not None
        assert memory.episodic is not None
        assert memory.semantic is not None
        assert memory.procedural is not None

    def test_before(self, memory):
        ctx = memory.before(person="alice_dev", message="hello")
        assert isinstance(ctx, Context)
        assert ctx.person == "alice"
        assert ctx.tokens_used > 0

    def test_after(self, memory):
        result = memory.after(
            person="alice",
            their_message="hello",
            response="Hey Alice! Good to hear from you.",
        )
        assert isinstance(result, AfterResult)
        assert result.signal.health > 0
        assert result.logged_message_id != ""

    def test_full_cycle(self, memory):
        """Test a complete before -> after cycle."""
        ctx = memory.before(person="alice", message="how's it going?")
        result = memory.after(
            person="alice",
            their_message="how's it going?",
            response="Pretty good! I've been thinking about that project we discussed.",
            trace_ids=ctx.trace_ids,
        )
        assert isinstance(result, AfterResult)

        # Second cycle should see conversation history
        ctx2 = memory.before(person="alice", message="what project?")
        assert ctx2.memories_loaded > 0

    def test_identity_resolution(self, memory):
        """Alias resolution works through the system."""
        ctx1 = memory.before(person="alice_dev", message="hi")
        ctx2 = memory.before(person="Alice", message="hi")
        assert ctx1.person == "alice"
        assert ctx2.person == "alice"

    def test_get_identity(self, memory):
        identity = memory.get_identity()
        assert "Test Identity" in identity

    def test_get_relationship(self, memory):
        # No relationship file yet
        assert memory.get_relationship("alice") is None

        # Create one via semantic store
        memory.semantic.add_fact("alice", "loves cats")
        rel = memory.get_relationship("alice")
        assert "cats" in rel

    def test_get_stats(self, memory):
        stats = memory.get_stats()
        assert isinstance(stats, MemoryStats)
        assert stats.episodic_count == 0  # fresh

        # After logging something
        memory.after(person="alice", their_message="hi", response="hello")
        stats2 = memory.get_stats()
        assert stats2.total_messages > 0

    def test_get_signal(self, memory):
        signal = memory.get_signal()
        assert "recent_health" in signal
        assert "trend" in signal

    def test_search(self, memory):
        memory.after(
            person="alice",
            their_message="I love programming in Python",
            response="Python is great for rapid prototyping!",
        )
        results = memory.search("Python programming")
        assert len(results) >= 1

    def test_decay_pass(self, memory):
        """Manual decay pass should not crash."""
        memory.decay_pass()

    def test_context_manager(self, config_with_soul):
        with MemorySystem(config=config_with_soul) as mem:
            ctx = mem.before(person="alice", message="hi")
            assert ctx.person == "alice"

    def test_token_budget_override(self, memory):
        ctx = memory.before(person="alice", message="hello", token_budget=2000)
        assert ctx.token_budget == 2000

    def test_multiple_people(self, memory):
        """Memory is person-scoped."""
        memory.after(person="alice", their_message="I like cats", response="Cool!")
        memory.after(person="bob", their_message="I like dogs", response="Nice!")

        ctx_alice = memory.before(person="alice", message="what do I like?")
        ctx_bob = memory.before(person="bob", message="what do I like?")

        # Both should have loaded their respective histories
        assert ctx_alice.person == "alice"
        assert ctx_bob.person == "bob"

    def test_signal_tracking_across_exchanges(self, memory):
        """Signal tracker accumulates across exchanges."""
        for i in range(3):
            memory.after(
                person="alice",
                their_message=f"message {i}",
                response=f"I think that's interesting. Let me consider what you said about message {i}.",
            )
        signal = memory.get_signal()
        assert signal["count"] == 3

    def test_boot(self, memory):
        """Boot returns project context and priming data."""
        boot_data = memory.boot()
        assert "soul" in boot_data
        assert "architecture_decisions" in boot_data
        assert "top_memories" in boot_data
        assert "preferences_summary" in boot_data
        assert "recent_journal" in boot_data
        assert "recent_sessions" in boot_data
        assert "workspace_items" in boot_data
        assert "signal_health" in boot_data
        assert "signal_trend" in boot_data

    def test_boot_with_content(self, memory):
        """Boot includes real data when available."""
        # Create some content first
        memory.after(
            person="alice",
            their_message="I love cats",
            response="Cats are wonderful companions.",
        )
        memory.journal.write("Test", "Reflection content")
        boot_data = memory.boot()
        assert boot_data["soul"] != ""  # SOUL.md exists from config_with_soul
        assert len(boot_data["recent_journal"]) >= 1

    def test_journal_subsystem(self, memory):
        """Journal is accessible through MemorySystem."""
        filename = memory.journal.write("Growth", "Learning new things.")
        entries = memory.journal.list_entries()
        assert len(entries) == 1
        content = memory.journal.read_entry(filename)
        assert "Learning new things" in content

    # Influence and injury subsystem tests removed in code-first pivot.
