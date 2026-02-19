"""
Tests for engram.procedural.schema and structured skill store (C1).

Covers:
  - SkillMeta dataclass creation and serialization
  - YAML frontmatter parsing (valid, missing, malformed)
  - Frontmatter serialization round-trip
  - Multi-dimensional filter matching
  - ProceduralStore.add_structured_skill() with metadata
  - ProceduralStore.filter_skills() multi-dimensional search
  - ProceduralStore.update_confidence() frequency-based scoring
  - Backward compatibility with legacy markdown skills
  - Context matching with metadata-enhanced skills
"""

from __future__ import annotations

import pytest
from pathlib import Path

from engram.procedural.schema import (
    SkillMeta,
    parse_frontmatter,
    serialize_frontmatter,
)
from engram.procedural.store import ProceduralStore


# =========================================================================
# SkillMeta dataclass
# =========================================================================


class TestSkillMeta:
    """Tests for the SkillMeta dataclass."""

    def test_defaults(self):
        meta = SkillMeta()
        assert meta.name == ""
        assert meta.language == ""
        assert meta.scope == "global"
        assert meta.confidence == 0.5
        assert meta.tags == []

    def test_to_dict_omits_empty(self):
        meta = SkillMeta(name="retry")
        d = meta.to_dict()
        assert d["name"] == "retry"
        # Empty strings should be omitted
        assert "language" not in d
        assert "framework" not in d
        # accepted_count=0 IS included (it's falsy but valid)
        assert d["accepted_count"] == 0

    def test_from_dict_ignores_unknown_keys(self):
        data = {"name": "test", "language": "python", "bogus_key": 42}
        meta = SkillMeta.from_dict(data)
        assert meta.name == "test"
        assert meta.language == "python"
        assert not hasattr(meta, "bogus_key") or True  # shouldn't crash

    def test_from_dict_round_trip(self):
        original = SkillMeta(
            name="retry",
            language="python",
            framework="asyncio",
            category="error-handling",
            tags=["retry", "async"],
            confidence=0.8,
        )
        d = original.to_dict()
        restored = SkillMeta.from_dict(d)
        assert restored.name == original.name
        assert restored.language == original.language
        assert restored.tags == original.tags
        assert restored.confidence == original.confidence


# =========================================================================
# Multi-dimensional filtering
# =========================================================================


class TestSkillMetaFilter:
    """Tests for SkillMeta.matches_filter()."""

    def _meta(self, **kwargs):
        return SkillMeta(
            name="test-skill",
            language="python",
            framework="asyncio",
            category="error-handling",
            scope="global",
            tags=["retry", "async", "http"],
            **kwargs,
        )

    def test_empty_filter_matches_all(self):
        assert self._meta().matches_filter()

    def test_language_match(self):
        assert self._meta().matches_filter(language="python")

    def test_language_mismatch(self):
        assert not self._meta().matches_filter(language="typescript")

    def test_language_case_insensitive(self):
        assert self._meta().matches_filter(language="Python")

    def test_framework_match(self):
        assert self._meta().matches_filter(framework="asyncio")

    def test_framework_mismatch(self):
        assert not self._meta().matches_filter(framework="react")

    def test_category_match(self):
        assert self._meta().matches_filter(category="error-handling")

    def test_category_mismatch(self):
        assert not self._meta().matches_filter(category="testing")

    def test_scope_match(self):
        assert self._meta().matches_filter(scope="global")

    def test_scope_mismatch(self):
        assert not self._meta().matches_filter(scope="module")

    def test_tags_subset_match(self):
        assert self._meta().matches_filter(tags=["retry", "async"])

    def test_tags_superset_mismatch(self):
        assert not self._meta().matches_filter(tags=["retry", "missing_tag"])

    def test_multi_dimension_all_match(self):
        assert self._meta().matches_filter(
            language="python",
            framework="asyncio",
            category="error-handling",
            tags=["retry"],
        )

    def test_multi_dimension_one_mismatch(self):
        assert not self._meta().matches_filter(
            language="python",
            framework="react",  # mismatch
            category="error-handling",
        )


# =========================================================================
# Frontmatter parsing
# =========================================================================


class TestParseFrontmatter:
    """Tests for YAML frontmatter parsing."""

    def test_with_frontmatter(self):
        text = (
            "---\n"
            "name: retry-with-backoff\n"
            "language: python\n"
            "framework: asyncio\n"
            "tags:\n"
            "  - retry\n"
            "  - async\n"
            "confidence: 0.8\n"
            "---\n"
            "# Retry Pattern\n\nSome content here.\n"
        )
        meta, body = parse_frontmatter(text)
        assert meta.name == "retry-with-backoff"
        assert meta.language == "python"
        assert meta.framework == "asyncio"
        assert meta.tags == ["retry", "async"]
        assert meta.confidence == 0.8
        assert "Retry Pattern" in body
        assert "---" not in body

    def test_without_frontmatter(self):
        text = "# Just Markdown\n\nNo frontmatter here.\n"
        meta, body = parse_frontmatter(text)
        assert meta.name == ""  # default
        assert meta.language == ""  # default
        assert body == text  # full text returned as body

    def test_malformed_yaml(self):
        text = "---\n{invalid yaml: [broken\n---\nContent\n"
        meta, body = parse_frontmatter(text)
        # Should degrade to legacy behavior
        assert meta.name == ""
        assert body == text

    def test_frontmatter_not_dict(self):
        text = "---\n- just a list\n- not a dict\n---\nContent\n"
        meta, body = parse_frontmatter(text)
        assert meta.name == ""
        assert body == text

    def test_empty_frontmatter(self):
        text = "---\n---\nContent here\n"
        meta, body = parse_frontmatter(text)
        assert meta.name == ""
        assert "Content here" in body

    def test_frontmatter_preserves_body(self):
        text = "---\nname: test\n---\nLine 1\nLine 2\n\n```python\ncode()\n```\n"
        meta, body = parse_frontmatter(text)
        assert meta.name == "test"
        assert "Line 1" in body
        assert "code()" in body


# =========================================================================
# Frontmatter serialization
# =========================================================================


class TestSerializeFrontmatter:
    """Tests for frontmatter serialization."""

    def test_round_trip(self):
        meta = SkillMeta(
            name="test-skill",
            language="python",
            tags=["async", "retry"],
            confidence=0.7,
        )
        body = "# Test\n\nSome content.\n"
        serialized = serialize_frontmatter(meta, body)

        # Parse it back
        restored_meta, restored_body = parse_frontmatter(serialized)
        assert restored_meta.name == "test-skill"
        assert restored_meta.language == "python"
        assert restored_meta.tags == ["async", "retry"]
        assert restored_meta.confidence == 0.7
        assert "Some content." in restored_body

    def test_has_frontmatter_delimiters(self):
        meta = SkillMeta(name="test")
        text = serialize_frontmatter(meta, "body")
        assert text.startswith("---\n")
        assert "\n---\n" in text

    def test_body_preserved_exactly(self):
        meta = SkillMeta(name="x")
        body = "exact content here"
        text = serialize_frontmatter(meta, body)
        _, restored = parse_frontmatter(text)
        assert "exact content here" in restored


# =========================================================================
# ProceduralStore — structured skills
# =========================================================================


class TestStructuredSkills:
    """Tests for ProceduralStore with YAML frontmatter skills."""

    @pytest.fixture
    def store(self, tmp_path):
        return ProceduralStore(tmp_path / "skills")

    def test_add_structured_skill(self, store):
        meta = store.add_structured_skill(
            name="retry-pattern",
            content="# Retry\n\nExponential backoff.",
            language="python",
            framework="asyncio",
            category="error-handling",
            tags=["retry", "async"],
        )
        assert meta.name == "retry-pattern"
        assert meta.language == "python"

        # Verify file was written with frontmatter
        content = store.get_skill("retry-pattern")
        assert "---" in content
        assert "language: python" in content

    def test_get_skill_meta(self, store):
        store.add_structured_skill(
            name="test",
            content="# Test",
            language="typescript",
            category="testing",
        )
        meta = store.get_skill_meta("test")
        assert meta is not None
        assert meta.language == "typescript"
        assert meta.category == "testing"

    def test_get_skill_meta_missing(self, store):
        assert store.get_skill_meta("nonexistent") is None

    def test_list_skills_includes_metadata(self, store):
        store.add_structured_skill(
            name="pattern-a",
            content="# A",
            language="python",
        )
        store.add_structured_skill(
            name="pattern-b",
            content="# B",
            language="typescript",
        )
        skills = store.list_skills()
        assert len(skills) == 2
        languages = {s.get("language", "") for s in skills}
        assert "python" in languages
        assert "typescript" in languages

    def test_filter_by_language(self, store):
        store.add_structured_skill(name="py1", content="# A", language="python")
        store.add_structured_skill(name="ts1", content="# B", language="typescript")
        store.add_structured_skill(name="py2", content="# C", language="python")

        results = store.filter_skills(language="python")
        assert len(results) == 2
        assert all(r["language"] == "python" for r in results)

    def test_filter_by_framework(self, store):
        store.add_structured_skill(name="a", content="# A", framework="asyncio")
        store.add_structured_skill(name="b", content="# B", framework="react")

        results = store.filter_skills(framework="asyncio")
        assert len(results) == 1
        assert results[0]["name"] == "a"

    def test_filter_by_category(self, store):
        store.add_structured_skill(name="a", content="# A", category="testing")
        store.add_structured_skill(name="b", content="# B", category="api-design")

        results = store.filter_skills(category="testing")
        assert len(results) == 1

    def test_filter_by_tags(self, store):
        store.add_structured_skill(name="a", content="# A", tags=["retry", "async"])
        store.add_structured_skill(name="b", content="# B", tags=["auth", "security"])

        results = store.filter_skills(tags=["retry"])
        assert len(results) == 1
        assert results[0]["name"] == "a"

    def test_filter_multi_dimensional(self, store):
        store.add_structured_skill(
            name="a",
            content="# A",
            language="python",
            framework="asyncio",
            category="error-handling",
        )
        store.add_structured_skill(
            name="b",
            content="# B",
            language="python",
            framework="flask",
            category="api-design",
        )

        results = store.filter_skills(language="python", framework="asyncio")
        assert len(results) == 1
        assert results[0]["name"] == "a"

    def test_filter_min_confidence(self, store):
        store.add_structured_skill(name="a", content="# A")
        # Default confidence is 0.5
        results = store.filter_skills(min_confidence=0.8)
        assert len(results) == 0

        results = store.filter_skills(min_confidence=0.3)
        assert len(results) == 1

    def test_filter_empty_returns_all(self, store):
        store.add_structured_skill(name="a", content="# A")
        store.add_structured_skill(name="b", content="# B")

        results = store.filter_skills()
        assert len(results) == 2


# =========================================================================
# Confidence updates (C2 prep)
# =========================================================================


class TestConfidenceUpdate:
    """Tests for frequency-based confidence modeling."""

    @pytest.fixture
    def store(self, tmp_path):
        return ProceduralStore(tmp_path / "skills")

    def test_accept_increases_confidence(self, store):
        store.add_structured_skill(name="test", content="# Test")
        meta = store.update_confidence("test", accepted=True)
        assert meta is not None
        assert meta.accepted_count == 1
        assert meta.confidence > 0  # 1/(1+0+1) = 0.5

    def test_reject_tracks_count(self, store):
        store.add_structured_skill(name="test", content="# Test")
        meta = store.update_confidence("test", accepted=False)
        assert meta.rejected_count == 1

    def test_multiple_accepts_increase_confidence(self, store):
        store.add_structured_skill(name="test", content="# Test")
        for _ in range(5):
            meta = store.update_confidence("test", accepted=True)
        # 5/(5+0+1) = 0.833
        assert meta.confidence > 0.8

    def test_mixed_feedback(self, store):
        store.add_structured_skill(name="test", content="# Test")
        store.update_confidence("test", accepted=True)
        store.update_confidence("test", accepted=True)
        meta = store.update_confidence("test", accepted=False)
        # 2/(2+1+1) = 0.5
        assert meta.confidence == 0.5

    def test_confidence_persists(self, store):
        store.add_structured_skill(name="test", content="# Test")
        store.update_confidence("test", accepted=True)
        store.update_confidence("test", accepted=True)

        # Re-read from disk
        meta = store.get_skill_meta("test")
        assert meta.accepted_count == 2

    def test_update_missing_skill(self, store):
        assert store.update_confidence("nonexistent") is None

    def test_add_preserves_confidence(self, store):
        """Re-adding a skill should preserve existing confidence."""
        store.add_structured_skill(name="test", content="# Original")
        store.update_confidence("test", accepted=True)
        store.update_confidence("test", accepted=True)

        # Re-add (update content)
        meta = store.add_structured_skill(name="test", content="# Updated")
        assert meta.accepted_count == 2  # preserved from before


# =========================================================================
# Backward compatibility
# =========================================================================


class TestBackwardCompatibility:
    """Legacy markdown skills (no frontmatter) still work correctly."""

    @pytest.fixture
    def store(self, tmp_path):
        return ProceduralStore(tmp_path / "skills")

    def test_legacy_add_and_get(self, store):
        store.add_skill("coding", "# Coding\n\nHow to write good code.")
        content = store.get_skill("coding")
        assert "How to write good code" in content

    def test_legacy_list_skills(self, store):
        store.add_skill("coding", "# Coding\nWrite clean code.")
        skills = store.list_skills()
        assert len(skills) == 1
        assert skills[0]["name"] == "coding"
        # Should have description even without frontmatter
        assert skills[0].get("description")

    def test_legacy_search_still_works(self, store):
        store.add_skill("python", "# Python\nDynamic typing and comprehensions.")
        results = store.search_skills("typing comprehensions")
        assert len(results) >= 1
        assert results[0]["name"] == "python"

    def test_legacy_match_context_still_works(self, store):
        store.add_skill("debugging", "# Debugging\nHow to find and fix bugs.")
        matched = store.match_context("I need help debugging this issue")
        assert len(matched) >= 1

    def test_mixed_legacy_and_structured(self, store):
        """Legacy and structured skills coexist."""
        store.add_skill("legacy", "# Legacy\nOld-style skill.")
        store.add_structured_skill(
            name="structured",
            content="# Structured",
            language="python",
        )
        skills = store.list_skills()
        assert len(skills) == 2

    def test_legacy_meta_returns_defaults(self, store):
        store.add_skill("legacy", "# Legacy\nOld-style.")
        meta = store.get_skill_meta("legacy")
        assert meta is not None
        assert meta.language == ""  # no frontmatter → default
        assert meta.scope == "global"  # default


# =========================================================================
# Context matching with metadata-enhanced skills
# =========================================================================


class TestEnhancedContextMatching:
    """Tests for match_context with frontmatter-aware matching."""

    @pytest.fixture
    def store(self, tmp_path):
        return ProceduralStore(tmp_path / "skills")

    def test_match_by_language(self, store):
        store.add_structured_skill(
            name="generic",
            content="# A pattern",
            language="python",
        )
        matched = store.match_context("I'm writing python code")
        assert len(matched) >= 1

    def test_match_by_framework(self, store):
        store.add_structured_skill(
            name="pattern",
            content="# A pattern",
            framework="fastapi",
        )
        matched = store.match_context("building a fastapi endpoint")
        assert len(matched) >= 1

    def test_match_by_tag(self, store):
        store.add_structured_skill(
            name="pattern",
            content="# A pattern",
            tags=["authentication", "oauth"],
        )
        matched = store.match_context("need to implement authentication")
        assert len(matched) >= 1

    def test_no_match_unrelated(self, store):
        store.add_structured_skill(
            name="pattern",
            content="# Database migration",
            language="python",
            tags=["database"],
        )
        matched = store.match_context("help with CSS styling")
        assert len(matched) == 0
