"""Tests for engram.trust — TrustGate, tiers, source gating, and enforcement.

Covers:
  - Tier enum and tier_from_name resolution
  - Source classification (privileged vs external)
  - AccessPolicy per-tier visibility
  - TrustGate.tier_for with source modifiers
  - TrustGate.policy_for
  - TrustGate.can_access
  - TrustGate.check_tool_access (source blocking + friend-required)
  - TrustGate.validate_promotion rules
  - TrustGate.filter_recall context gating
  - TrustGate.ensure_core_person
  - Integration: before-pipeline trust filtering
  - Integration: after-pipeline skip_persistence for strangers
  - Integration: server tool gating from external sources
"""

import json
import pytest

from engram.trust import (
    ACCESS,
    EXTERNAL_SOURCES,
    FRIEND_REQUIRED_TOOLS,
    PRIVILEGED_SOURCES,
    SOURCE_BLOCKED_TOOLS,
    AccessPolicy,
    Tier,
    TrustGate,
    is_privileged_source,
    tier_from_name,
)
from engram.system import MemorySystem
import engram.server as server_mod


# ===========================================================================
# Tier enum
# ===========================================================================


class TestTierEnum:
    def test_ordering(self):
        assert (
            Tier.CORE
            < Tier.INNER_CIRCLE
            < Tier.FRIEND
            < Tier.ACQUAINTANCE
            < Tier.STRANGER
        )

    def test_values(self):
        assert Tier.CORE == 0
        assert Tier.STRANGER == 4

    def test_min_is_core(self):
        assert min(Tier) == Tier.CORE

    def test_max_is_stranger(self):
        assert max(Tier) == Tier.STRANGER


class TestTierFromName:
    def test_all_known_tiers(self):
        assert tier_from_name("core") == Tier.CORE
        assert tier_from_name("inner_circle") == Tier.INNER_CIRCLE
        assert tier_from_name("friend") == Tier.FRIEND
        assert tier_from_name("acquaintance") == Tier.ACQUAINTANCE
        assert tier_from_name("stranger") == Tier.STRANGER

    def test_case_insensitive(self):
        assert tier_from_name("CORE") == Tier.CORE
        assert tier_from_name("Friend") == Tier.FRIEND

    def test_spaces_to_underscores(self):
        assert tier_from_name("inner circle") == Tier.INNER_CIRCLE

    def test_unknown_defaults_to_stranger(self):
        assert tier_from_name("overlord") == Tier.STRANGER
        assert tier_from_name("") == Tier.STRANGER


# ===========================================================================
# Source classification
# ===========================================================================


class TestSourceClassification:
    def test_privileged_sources(self):
        for source in ["direct", "opencode", "cli"]:
            assert is_privileged_source(source), f"{source} should be privileged"

    def test_external_sources(self):
        for source in ["discord", "voice", "api", "openclaw"]:
            assert not is_privileged_source(source), (
                f"{source} should NOT be privileged"
            )

    def test_unknown_source_is_external(self):
        assert not is_privileged_source("telegram")

    def test_case_insensitive(self):
        assert is_privileged_source("Direct")
        assert is_privileged_source("CLI")

    def test_sets_are_disjoint(self):
        assert PRIVILEGED_SOURCES.isdisjoint(EXTERNAL_SOURCES)


# ===========================================================================
# Access policies
# ===========================================================================


class TestAccessPolicies:
    def test_core_sees_everything(self):
        p = ACCESS[Tier.CORE]
        assert p.can_see_soul
        assert p.can_see_own_relationship
        assert p.can_see_others_relationships
        assert p.can_see_preferences
        assert p.can_see_boundaries
        assert p.can_see_contradictions
        assert p.can_see_injuries
        assert p.can_see_journal
        assert p.can_see_influence_log
        assert p.memory_persistent
        assert p.can_modify_soul
        assert p.can_modify_trust

    def test_stranger_sees_nothing(self):
        p = ACCESS[Tier.STRANGER]
        assert not p.can_see_soul
        assert not p.can_see_own_relationship
        assert not p.can_see_preferences
        assert not p.can_see_boundaries
        assert not p.memory_persistent
        assert not p.personal_topics

    def test_friend_sees_own_relationship_and_prefs(self):
        p = ACCESS[Tier.FRIEND]
        assert not p.can_see_soul  # Friends don't see SOUL.md
        assert p.can_see_own_relationship
        assert not p.can_see_others_relationships
        assert p.can_see_preferences
        assert p.can_see_boundaries
        assert not p.can_see_contradictions
        assert not p.can_see_injuries
        assert p.memory_persistent

    def test_acquaintance_minimal(self):
        p = ACCESS[Tier.ACQUAINTANCE]
        assert not p.can_see_soul
        assert p.can_see_own_relationship  # Can see their own relationship file
        assert not p.can_see_preferences
        assert not p.personal_topics
        assert p.memory_persistent  # Acquaintances DO get memory persistence

    def test_inner_circle_deep_access(self):
        p = ACCESS[Tier.INNER_CIRCLE]
        assert p.can_see_soul
        assert p.can_see_others_relationships
        assert p.can_see_contradictions
        assert p.can_see_injuries
        assert p.can_see_journal
        assert not p.can_see_influence_log  # Only core
        assert not p.can_modify_soul
        assert not p.can_modify_trust

    def test_all_tiers_have_policies(self):
        for tier in Tier:
            assert tier in ACCESS, f"No policy for {tier.name}"

    def test_policy_is_frozen(self):
        p = ACCESS[Tier.CORE]
        with pytest.raises(AttributeError):
            p.can_see_soul = False  # type: ignore[misc]


# ===========================================================================
# TrustGate
# ===========================================================================


@pytest.fixture
def gate(config_with_soul):
    """TrustGate with pre-populated trust data."""
    from engram.semantic.store import SemanticStore

    semantic = SemanticStore(
        semantic_dir=config_with_soul.semantic_dir,
        soul_dir=config_with_soul.soul_dir,
    )
    return TrustGate(semantic=semantic, core_person="tester")


class TestTrustGateTierResolution:
    def test_core_person(self, gate):
        assert gate.tier_for("tester") == Tier.CORE

    def test_friend(self, gate):
        assert gate.tier_for("alice") == Tier.FRIEND

    def test_acquaintance(self, gate):
        assert gate.tier_for("bob") == Tier.ACQUAINTANCE

    def test_unknown_is_stranger(self, gate):
        assert gate.tier_for("nobody") == Tier.STRANGER


class TestTrustGateSourceModifier:
    def test_external_demotes_friend_to_acquaintance(self, gate):
        assert gate.tier_for("alice", source="discord") == Tier.ACQUAINTANCE

    def test_external_demotes_acquaintance_to_stranger(self, gate):
        assert gate.tier_for("bob", source="discord") == Tier.STRANGER

    def test_core_is_exempt_from_demotion(self, gate):
        assert gate.tier_for("tester", source="discord") == Tier.CORE

    def test_stranger_stays_stranger(self, gate):
        # Can't demote past stranger
        assert gate.tier_for("nobody", source="api") == Tier.STRANGER

    def test_direct_source_no_demotion(self, gate):
        assert gate.tier_for("alice", source="direct") == Tier.FRIEND


class TestTrustGateCanAccess:
    def test_core_can_access_everything(self, gate):
        assert gate.can_access("tester", "core")
        assert gate.can_access("tester", "stranger")

    def test_stranger_cannot_access_friend(self, gate):
        assert not gate.can_access("nobody", "friend")

    def test_friend_can_access_friend(self, gate):
        assert gate.can_access("alice", "friend")

    def test_friend_cannot_access_core(self, gate):
        assert not gate.can_access("alice", "core")


# ===========================================================================
# Tool access gating
# ===========================================================================


class TestToolAccess:
    def test_privileged_source_always_allowed(self, gate):
        for tool in SOURCE_BLOCKED_TOOLS:
            assert gate.check_tool_access(tool, "nobody", source="direct") is None

    def test_source_blocked_from_external(self, gate):
        for tool in SOURCE_BLOCKED_TOOLS:
            result = gate.check_tool_access(tool, "tester", source="discord")
            assert result is not None, f"{tool} should be blocked from discord"
            assert "not available" in result

    def test_friend_required_allowed_for_friend(self, gate):
        for tool in FRIEND_REQUIRED_TOOLS:
            # Alice is friend from direct source
            assert gate.check_tool_access(tool, "alice", source="direct") is None

    def test_friend_required_blocked_for_stranger_external(self, gate):
        for tool in FRIEND_REQUIRED_TOOLS:
            result = gate.check_tool_access(tool, "nobody", source="discord")
            assert result is not None, f"{tool} should block stranger from discord"

    def test_friend_required_blocked_for_demoted_friend(self, gate):
        # Alice is friend, but from discord becomes acquaintance
        for tool in FRIEND_REQUIRED_TOOLS:
            result = gate.check_tool_access(tool, "alice", source="discord")
            assert result is not None, (
                f"{tool} should block alice from discord (demoted)"
            )

    def test_unknown_tool_allowed(self, gate):
        assert (
            gate.check_tool_access("engram_search", "nobody", source="discord") is None
        )


# ===========================================================================
# Promotion validation
# ===========================================================================


class TestPromotionValidation:
    def test_auto_promote_stranger_to_friend(self, gate):
        assert gate.validate_promotion("nobody", "friend", promoted_by="auto") is None

    def test_auto_promote_acquaintance_to_friend(self, gate):
        assert gate.validate_promotion("bob", "friend", promoted_by="auto") is None

    def test_auto_promote_to_inner_circle_blocked(self, gate):
        result = gate.validate_promotion("bob", "inner_circle", promoted_by="auto")
        assert result is not None
        assert "core tier" in result.lower() or "explicit approval" in result.lower()

    def test_core_person_can_promote_to_inner_circle(self, gate):
        result = gate.validate_promotion("bob", "inner_circle", promoted_by="tester")
        assert result is None

    def test_non_core_cannot_promote_to_inner_circle(self, gate):
        result = gate.validate_promotion("bob", "inner_circle", promoted_by="alice")
        assert result is not None

    def test_cannot_promote_to_same_tier(self, gate):
        result = gate.validate_promotion("alice", "friend", promoted_by="tester")
        assert result is not None
        assert "already at" in result.lower()

    def test_cannot_promote_to_lower_tier(self, gate):
        result = gate.validate_promotion("alice", "acquaintance", promoted_by="tester")
        assert result is not None

    def test_external_source_blocks_promotion(self, gate):
        result = gate.validate_promotion(
            "bob", "friend", promoted_by="tester", source="discord"
        )
        assert result is not None
        assert "external" in result.lower()

    def test_promote_to_core_requires_core_person(self, gate):
        result = gate.validate_promotion("alice", "core", promoted_by="alice")
        assert result is not None


# ===========================================================================
# Recall filtering
# ===========================================================================


class TestRecallFiltering:
    def test_core_can_recall_everything(self, gate):
        assert gate.filter_recall("tester", "identity") is None
        assert gate.filter_recall("tester", "preferences") is None
        assert gate.filter_recall("tester", "contradictions") is None
        assert (
            gate.filter_recall("tester", "relationship", target_person="alice") is None
        )

    def test_stranger_blocked_from_identity(self, gate):
        result = gate.filter_recall("nobody", "identity")
        assert result is not None

    def test_stranger_blocked_from_preferences(self, gate):
        result = gate.filter_recall("nobody", "preferences")
        assert result is not None

    def test_friend_can_see_own_relationship(self, gate):
        assert (
            gate.filter_recall("alice", "relationship", target_person="alice") is None
        )

    def test_friend_cannot_see_others_relationship(self, gate):
        result = gate.filter_recall("alice", "relationship", target_person="bob")
        assert result is not None

    def test_inner_circle_can_see_others_relationships(self, gate):
        # Manually promote someone to inner_circle for this test
        gate._semantic.update_trust("alice", "inner_circle", "test promotion")
        assert gate.filter_recall("alice", "relationship", target_person="bob") is None

    def test_friend_blocked_from_contradictions(self, gate):
        result = gate.filter_recall("alice", "contradictions")
        assert result is not None

    def test_stranger_blocked_from_boundaries(self, gate):
        result = gate.filter_recall("nobody", "boundaries")
        assert result is not None

    def test_friend_can_see_preferences(self, gate):
        assert gate.filter_recall("alice", "preferences") is None

    def test_external_source_demotes(self, gate):
        # Alice from discord → acquaintance → no preferences
        result = gate.filter_recall("alice", "preferences", source="discord")
        assert result is not None


# ===========================================================================
# Core person registration
# ===========================================================================


class TestCorePerson:
    def test_ensure_core_person_registers(self, config_with_soul):
        from engram.semantic.store import SemanticStore

        semantic = SemanticStore(
            semantic_dir=config_with_soul.semantic_dir,
            soul_dir=config_with_soul.soul_dir,
        )
        gate = TrustGate(semantic=semantic, core_person="newowner")
        gate.ensure_core_person()

        info = semantic.check_trust("newowner")
        assert info["tier"] == "core"

    def test_no_core_person_is_noop(self, config_with_soul):
        from engram.semantic.store import SemanticStore

        semantic = SemanticStore(
            semantic_dir=config_with_soul.semantic_dir,
            soul_dir=config_with_soul.soul_dir,
        )
        gate = TrustGate(semantic=semantic, core_person="")
        gate.ensure_core_person()  # Should not crash

    def test_core_person_idempotent(self, config_with_soul):
        from engram.semantic.store import SemanticStore

        semantic = SemanticStore(
            semantic_dir=config_with_soul.semantic_dir,
            soul_dir=config_with_soul.soul_dir,
        )
        gate = TrustGate(semantic=semantic, core_person="tester")
        gate.ensure_core_person()
        gate.ensure_core_person()  # Second call should not crash

        info = semantic.check_trust("tester")
        assert info["tier"] == "core"


# ===========================================================================
# Integration: before-pipeline trust filtering
# ===========================================================================


class TestBeforePipelineTrust:
    def test_stranger_gets_empty_context(self, config_with_soul):
        system = MemorySystem(config=config_with_soul)
        try:
            # Add some content that should be hidden from strangers
            system.semantic.update_preferences("cats", "like", "they purr")
            system.semantic.add_boundary("Identity", "Don't rewrite my soul")

            ctx = system.before(person="unknown_stranger", message="hello")
            # Stranger should not see preferences or boundaries
            assert "cats" not in ctx.text
            assert "Don't rewrite" not in ctx.text
            # But should see trust info
            assert "stranger" in ctx.text.lower()
        finally:
            system.close()

    def test_friend_sees_preferences_and_boundaries(self, config_with_soul):
        system = MemorySystem(config=config_with_soul)
        try:
            system.semantic.update_preferences("cats", "like", "they purr")
            system.semantic.add_boundary("Identity", "Don't rewrite my soul")

            ctx = system.before(person="alice", message="hello")
            assert "cats" in ctx.text
            assert "Don't rewrite" in ctx.text
        finally:
            system.close()

    def test_friend_does_not_see_soul(self, config_with_soul):
        system = MemorySystem(config=config_with_soul)
        try:
            ctx = system.before(person="alice", message="hello")
            # Friends don't see SOUL.md content
            assert "Test Identity" not in ctx.text
        finally:
            system.close()

    def test_core_person_sees_soul(self, config_with_soul):
        system = MemorySystem(config=config_with_soul)
        try:
            ctx = system.before(person="tester", message="hello")
            assert "Test Identity" in ctx.text
        finally:
            system.close()


# ===========================================================================
# Integration: after-pipeline skip_persistence
# ===========================================================================


class TestAfterPipelineTrust:
    def test_stranger_messages_not_persisted(self, config_with_soul):
        system = MemorySystem(config=config_with_soul)
        try:
            system.after(
                person="stranger_person",
                their_message="hi",
                response="hello",
            )
            msgs = system.episodic.get_recent_messages(person="stranger_person")
            assert len(msgs) == 0
        finally:
            system.close()

    def test_friend_messages_persisted(self, config_with_soul):
        system = MemorySystem(config=config_with_soul)
        try:
            system.after(
                person="alice",
                their_message="hi",
                response="hello",
            )
            msgs = system.episodic.get_recent_messages(person="alice")
            assert len(msgs) > 0
        finally:
            system.close()

    def test_stranger_still_gets_signal_measured(self, config_with_soul):
        system = MemorySystem(config=config_with_soul)
        try:
            result = system.after(
                person="stranger_person",
                their_message="hi",
                response="hello there",
            )
            # Signal should still be measured even if persistence is skipped
            assert result.signal.health > 0
        finally:
            system.close()


# ===========================================================================
# Integration: server tool gating from external sources
# ===========================================================================


class TestServerTrustGating:
    @pytest.fixture
    def server_system(self, config_with_soul):
        """Wire up server with external source for testing."""
        system = MemorySystem(config=config_with_soul)
        old_system = server_mod._system
        old_source = server_mod._current_source
        old_person = server_mod._current_person
        server_mod._system = system
        yield system
        server_mod._system = old_system
        server_mod._current_source = old_source
        server_mod._current_person = old_person
        system.close()

    def test_blocked_tool_from_discord(self, server_system):
        server_mod._current_source = "discord"
        server_mod._current_person = "alice"

        result = server_mod.engram_trust_promote(
            person="bob", new_tier="friend", reason="test"
        )
        data = json.loads(result)
        assert data.get("error") is True or "not available" in data.get("message", "")

    def test_allowed_tool_from_direct(self, server_system):
        server_mod._current_source = "direct"
        server_mod._current_person = "tester"

        result = server_mod.engram_trust_check(person="alice")
        data = json.loads(result)
        assert "tier" in data

    def test_recall_identity_blocked_for_stranger(self, server_system):
        server_mod._current_source = "direct"
        server_mod._current_person = "nobody"

        result = server_mod.engram_recall(what="identity")
        data = json.loads(result)
        assert data.get("error") is True

    def test_recall_identity_allowed_for_core(self, server_system):
        server_mod._current_source = "direct"
        server_mod._current_person = "tester"

        result = server_mod.engram_recall(what="identity")
        assert "Test Identity" in result

    def test_friend_required_tool_blocked_for_stranger_discord(self, server_system):
        server_mod._current_source = "discord"
        server_mod._current_person = "nobody"

        result = server_mod.engram_add_fact(person="nobody", fact="test fact")
        data = json.loads(result)
        assert data.get("error") is True

    def test_before_sets_source_and_person(self, server_system):
        server_mod.engram_before(person="alice", message="hello", source="discord")
        assert server_mod._current_source == "discord"
        assert server_mod._current_person == "alice"
