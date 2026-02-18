"""Tests for the code-first SOUL.md philosophy generator.

Verifies template construction, markdown generation, file I/O,
and round-trip parsing.
"""

import pytest
from pathlib import Path

from engram.soul import (
    SoulTemplate,
    generate_soul,
    write_soul,
    load_template_from_soul,
    DEFAULT_PHILOSOPHY,
    DEFAULT_REVIEW_CHECKLIST,
    DEFAULT_FORBIDDEN,
)


# ---------------------------------------------------------------------------
# SoulTemplate tests
# ---------------------------------------------------------------------------


class TestSoulTemplate:
    """Test template construction and defaults."""

    def test_default_template(self):
        t = SoulTemplate()
        assert t.name == "engram"
        assert len(t.philosophy) == len(DEFAULT_PHILOSOPHY)
        assert len(t.review_checklist) == len(DEFAULT_REVIEW_CHECKLIST)
        assert len(t.forbidden) == len(DEFAULT_FORBIDDEN)

    def test_custom_template(self):
        t = SoulTemplate(
            name="my-project",
            philosophy=["DRY", "KISS"],
            proficiency={"Python": "expert", "Rust": "learning"},
            patterns=["Repository pattern"],
            review_checklist=["All functions typed"],
            forbidden=["eval()"],
        )
        assert t.name == "my-project"
        assert len(t.philosophy) == 2
        assert t.proficiency["Python"] == "expert"

    def test_default_lists_are_independent(self):
        """Modifying one template's defaults shouldn't affect another."""
        t1 = SoulTemplate()
        t2 = SoulTemplate()
        t1.philosophy.append("Extra principle")
        assert len(t1.philosophy) != len(t2.philosophy)


# ---------------------------------------------------------------------------
# Generation tests
# ---------------------------------------------------------------------------


class TestGenerateSoul:
    """Test markdown generation from templates."""

    def test_default_generation(self):
        text = generate_soul()
        assert "# engram" in text
        assert "## Philosophy" in text
        assert "Test-First" in text
        assert "## Code Review Checklist" in text
        assert "## Forbidden Constructs" in text

    def test_custom_name(self):
        t = SoulTemplate(name="awesome-project")
        text = generate_soul(t)
        assert "# awesome-project" in text

    def test_philosophy_items(self):
        t = SoulTemplate(philosophy=["DRY", "YAGNI", "KISS"])
        text = generate_soul(t)
        assert "**DRY**" in text
        assert "**YAGNI**" in text
        assert "**KISS**" in text

    def test_proficiency_table(self):
        t = SoulTemplate(proficiency={"Python": "expert", "Go": "proficient"})
        text = generate_soul(t)
        assert "## Proficiency" in text
        assert "| Python | expert |" in text
        assert "| Go | proficient |" in text

    def test_no_proficiency_skips_section(self):
        t = SoulTemplate(proficiency={})
        text = generate_soul(t)
        assert "## Proficiency" not in text

    def test_patterns(self):
        t = SoulTemplate(patterns=["Repository pattern", "Event sourcing"])
        text = generate_soul(t)
        assert "## Architectural Patterns" in text
        assert "Repository pattern" in text

    def test_no_patterns_skips_section(self):
        t = SoulTemplate(patterns=[])
        text = generate_soul(t)
        assert "## Architectural Patterns" not in text

    def test_review_checklist_uses_checkboxes(self):
        t = SoulTemplate(review_checklist=["All functions typed"])
        text = generate_soul(t)
        assert "- [ ] All functions typed" in text

    def test_forbidden_uses_strikethrough(self):
        t = SoulTemplate(forbidden=["eval()"])
        text = generate_soul(t)
        assert "~~eval()~~" in text

    def test_custom_sections(self):
        t = SoulTemplate(
            custom_sections={
                "Team Conventions": "We use feature branches.\nPRs require two reviewers.",
            }
        )
        text = generate_soul(t)
        assert "## Team Conventions" in text
        assert "feature branches" in text

    def test_none_template_uses_defaults(self):
        text = generate_soul(None)
        assert "Test-First" in text


# ---------------------------------------------------------------------------
# File I/O tests
# ---------------------------------------------------------------------------


class TestWriteSoul:
    """Test writing SOUL.md to filesystem."""

    def test_writes_file(self, tmp_path):
        path = write_soul(soul_dir=tmp_path)
        assert path.exists()
        assert path.name == "SOUL.md"
        content = path.read_text(encoding="utf-8")
        assert "## Philosophy" in content

    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "deep" / "nested"
        path = write_soul(soul_dir=new_dir)
        assert path.exists()

    def test_no_overwrite_by_default(self, tmp_path):
        write_soul(soul_dir=tmp_path)
        with pytest.raises(FileExistsError):
            write_soul(soul_dir=tmp_path)

    def test_overwrite_when_requested(self, tmp_path):
        write_soul(soul_dir=tmp_path)
        path = write_soul(soul_dir=tmp_path, overwrite=True)
        assert path.exists()

    def test_custom_template(self, tmp_path):
        t = SoulTemplate(name="test-project", philosophy=["KISS"])
        path = write_soul(template=t, soul_dir=tmp_path)
        content = path.read_text(encoding="utf-8")
        assert "test-project" in content
        assert "KISS" in content


# ---------------------------------------------------------------------------
# Round-trip parsing tests
# ---------------------------------------------------------------------------


class TestLoadTemplate:
    """Test parsing an existing SOUL.md back into a template."""

    def test_round_trip(self, tmp_path):
        """Write a SOUL.md and parse it back — key fields should survive."""
        original = SoulTemplate(
            name="round-trip-test",
            philosophy=["Test-First", "KISS"],
            proficiency={"Python": "expert", "Go": "learning"},
            patterns=["Repository pattern", "DI"],
            review_checklist=["Functions typed", "Docstrings present"],
            forbidden=["eval()", "import *"],
        )
        path = write_soul(template=original, soul_dir=tmp_path)
        loaded = load_template_from_soul(path)

        # Name extraction is best-effort
        assert "round" in loaded.name.lower()

        # Philosophy items should survive (may have bold markers stripped)
        assert len(loaded.philosophy) == 2

        # Proficiency table
        assert loaded.proficiency.get("Python") == "expert"
        assert loaded.proficiency.get("Go") == "learning"

        # Patterns
        assert len(loaded.patterns) == 2

    def test_nonexistent_file(self, tmp_path):
        """Loading from nonexistent file returns default template."""
        t = load_template_from_soul(tmp_path / "nope.md")
        assert isinstance(t, SoulTemplate)
        assert t.name == "engram"  # Default

    def test_empty_file(self, tmp_path):
        """Loading from empty file returns default template."""
        path = tmp_path / "SOUL.md"
        path.write_text("", encoding="utf-8")
        t = load_template_from_soul(path)
        assert isinstance(t, SoulTemplate)
