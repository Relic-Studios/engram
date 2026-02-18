"""Tests for engram.procedural.store."""

import pytest


class TestProceduralStore:
    def test_list_empty(self, procedural):
        assert procedural.list_skills() == []

    def test_add_and_get(self, procedural):
        procedural.add_skill("coding", "# Coding\n\nHow to write good code.")
        content = procedural.get_skill("coding")
        assert "How to write good code" in content

    def test_list_skills(self, procedural):
        procedural.add_skill("coding", "# Coding\nWrite clean code.")
        procedural.add_skill("testing", "# Testing\nWrite good tests.")
        skills = procedural.list_skills()
        assert len(skills) == 2
        names = {s["name"] for s in skills}
        assert "coding" in names
        assert "testing" in names

    def test_get_missing(self, procedural):
        assert procedural.get_skill("nonexistent") is None

    def test_search_skills(self, procedural):
        procedural.add_skill(
            "python", "# Python\nDynamic typing and list comprehensions."
        )
        procedural.add_skill("rust", "# Rust\nOwnership and borrowing.")
        results = procedural.search_skills("typing comprehensions")
        assert len(results) >= 1
        assert results[0]["name"] == "python"

    def test_search_empty(self, procedural):
        assert procedural.search_skills("anything") == []

    def test_match_context(self, procedural):
        procedural.add_skill("debugging", "# Debugging\nHow to find and fix bugs.")
        matched = procedural.match_context("I need help debugging this issue")
        assert len(matched) >= 1
        assert "Debugging" in matched[0]

    def test_match_context_no_match(self, procedural):
        procedural.add_skill("cooking", "# Cooking\nHow to make pasta.")
        matched = procedural.match_context("help me with my code")
        assert len(matched) == 0

    def test_sanitized_name(self, procedural):
        procedural.add_skill("my skill/name!", "content")
        skills = procedural.list_skills()
        assert len(skills) == 1
        # Name should be sanitized (no slashes or special chars)
        assert "/" not in skills[0]["name"]
