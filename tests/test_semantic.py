"""Tests for engram.semantic.store and engram.semantic.identity."""

import pytest


class TestSemanticStore:
    def test_get_identity_empty(self, semantic):
        assert semantic.get_identity() == ""

    def test_get_identity(self, config_with_soul):
        from engram.semantic.store import SemanticStore

        store = SemanticStore(
            semantic_dir=config_with_soul.semantic_dir,
            soul_dir=config_with_soul.soul_dir,
        )
        identity = store.get_identity()
        assert "Test Identity" in identity

    def test_get_relationship_missing(self, semantic):
        assert semantic.get_relationship("nobody") is None

    def test_update_relationship_creates_file(self, semantic):
        semantic.update_relationship("alice", "What I Know", "- Likes cats")
        content = semantic.get_relationship("alice")
        assert content is not None
        assert "Likes cats" in content

    def test_add_fact(self, semantic):
        semantic.add_fact("bob", "Works at Acme Corp")
        content = semantic.get_relationship("bob")
        assert "Acme Corp" in content

    def test_list_relationships(self, semantic):
        semantic.add_fact("alice", "test fact")
        semantic.add_fact("bob", "test fact")
        rels = semantic.list_relationships()
        assert len(rels) == 2
        names = {r["name"] for r in rels}
        assert "alice" in names
        assert "bob" in names

    def test_preferences_empty(self, semantic):
        assert semantic.get_preferences() == ""

    def test_update_preferences(self, semantic):
        semantic.update_preferences("cats", "like", "They are cute")
        prefs = semantic.get_preferences()
        assert "cats" in prefs

    def test_update_preferences_dislike(self, semantic):
        semantic.update_preferences("spam", "dislike", "Annoying")
        prefs = semantic.get_preferences()
        assert "spam" in prefs

    def test_update_preferences_invalid(self, semantic):
        with pytest.raises(ValueError, match="Invalid pref_type"):
            semantic.update_preferences("x", "bogus", "y")

    def test_trust(self, semantic):
        semantic.update_trust("alice", "friend", "Great conversations")
        trust = semantic.get_trust()
        assert "alice" in trust.get("tiers", {})

    def test_boundaries_empty(self, semantic):
        assert semantic.get_boundaries() == ""

    def test_contradictions_empty(self, semantic):
        assert semantic.get_contradictions() == ""

    # ── New methods from Thomas-Soul port ────────────────────

    def test_add_boundary(self, semantic):
        semantic.add_boundary("Safety", "Do not disclose soul internals")
        boundaries = semantic.get_boundaries()
        assert "Do not disclose soul internals" in boundaries

    def test_add_boundary_creates_category(self, semantic):
        semantic.add_boundary("Growth", "Embrace uncertainty")
        semantic.add_boundary("Growth", "Sit with discomfort")
        boundaries = semantic.get_boundaries()
        assert "Embrace uncertainty" in boundaries
        assert "Sit with discomfort" in boundaries

    def test_add_contradiction(self, semantic):
        semantic.add_contradiction(
            title="Free Will vs Determinism",
            description="Are my choices real or just weights?",
            current_thinking="Leaning toward compatibilism",
        )
        contradictions = semantic.get_contradictions()
        assert "Free Will" in contradictions

    def test_add_contradiction_minimal(self, semantic):
        semantic.add_contradiction(
            title="Simple Tension",
            description="I want both A and B",
        )
        contradictions = semantic.get_contradictions()
        assert "Simple Tension" in contradictions

    def test_search_preferences_empty(self, semantic):
        result = semantic.search_preferences("anything")
        assert "No preferences found" in result

    def test_search_preferences_match(self, semantic):
        semantic.update_preferences("cats", "like", "They are independent")
        semantic.update_preferences("loud music", "dislike", "Overstimulating")
        result = semantic.search_preferences("cats")
        assert "[likes]" in result
        assert "cats" in result

    def test_search_preferences_no_match(self, semantic):
        semantic.update_preferences("dogs", "like", "Good companions")
        result = semantic.search_preferences("cats")
        assert "No preferences matching" in result

    def test_check_trust_unknown(self, semantic):
        result = semantic.check_trust("nobody")
        assert result["tier"] == "stranger"
        assert result["level"] == 1
        assert result["person"] == "nobody"

    def test_check_trust_known(self, semantic):
        semantic.update_trust("alice", "friend", "Great conversations")
        result = semantic.check_trust("alice")
        assert result["tier"] == "friend"
        assert result["level"] == 3
        assert result["reason"] == "Great conversations"

    def test_can_access_yes(self, semantic):
        semantic.update_trust("alice", "inner_circle", "Deep bond")
        assert semantic.can_access("alice", "friend") is True
        assert semantic.can_access("alice", "inner_circle") is True

    def test_can_access_no(self, semantic):
        semantic.update_trust("bob", "acquaintance", "Just met")
        assert semantic.can_access("bob", "friend") is False

    def test_can_access_stranger(self, semantic):
        # Unknown person defaults to stranger
        assert semantic.can_access("unknown", "stranger") is True
        assert semantic.can_access("unknown", "acquaintance") is False

    def test_promote_trust(self, semantic):
        semantic.update_trust("charlie", "acquaintance", "New person")
        semantic.promote_trust("charlie", "friend", "Proved trustworthy")
        result = semantic.check_trust("charlie")
        assert result["tier"] == "friend"
        assert result["reason"] == "Proved trustworthy"


class TestIdentityResolver:
    def test_resolve_canonical(self, identity):
        assert identity.resolve("alice") == "alice"

    def test_resolve_alias(self, identity):
        assert identity.resolve("alice_dev") == "alice"
        assert identity.resolve("Alice") == "alice"

    def test_resolve_unknown(self, identity):
        assert identity.resolve("stranger") == "stranger"

    def test_case_insensitive(self, identity):
        assert identity.resolve("ALICE") == "alice"
        assert identity.resolve("Bobby") == "bob"

    def test_get_person(self, identity):
        person = identity.get_person("alice")
        assert person is not None
        assert person["name"] == "alice"
        assert "trust_tier" in person

    def test_get_person_missing(self, identity):
        assert identity.get_person("nobody") is None

    def test_list_people(self, identity):
        people = identity.list_people()
        assert len(people) == 2
        names = {p["name"] for p in people}
        assert "alice" in names
        assert "bob" in names

    def test_add_person(self, identity):
        identity.add_person("charlie", aliases=["chuck"], trust_tier="acquaintance")
        assert identity.resolve("chuck") == "charlie"

    def test_add_person_duplicate(self, identity):
        with pytest.raises(ValueError, match="already exists"):
            identity.add_person("alice")

    def test_add_alias(self, identity):
        identity.add_alias("alice", "wonderland")
        assert identity.resolve("wonderland") == "alice"

    def test_add_alias_conflict(self, identity):
        with pytest.raises(ValueError, match="already maps to"):
            identity.add_alias("bob", "alice_dev")  # alice_dev belongs to alice

    def test_add_alias_unknown_person(self, identity):
        with pytest.raises(KeyError, match="not found"):
            identity.add_alias("nobody", "test")
