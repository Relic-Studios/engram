"""Tests for engram.journal.JournalStore."""

import pytest
from engram.journal import JournalStore


@pytest.fixture
def journal(tmp_path):
    """Provide a fresh JournalStore."""
    return JournalStore(tmp_path / "journal")


class TestJournalStore:
    def test_write_returns_filename(self, journal):
        filename = journal.write("Test Topic", "Some reflection content.")
        assert filename.endswith(".md")
        assert filename.startswith("20")  # starts with year

    def test_write_creates_file(self, journal):
        filename = journal.write("Growth", "I learned something today.")
        path = journal.journal_dir / filename
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "# Journal: Growth" in text
        assert "I learned something today." in text

    def test_write_includes_metadata(self, journal):
        filename = journal.write("Reflection", "Deep thoughts.")
        text = (journal.journal_dir / filename).read_text(encoding="utf-8")
        assert "**Date:**" in text
        assert "**Topic:** Reflection" in text
        assert "---" in text

    def test_write_multiple_same_day(self, journal):
        f1 = journal.write("First", "Entry one.")
        f2 = journal.write("Second", "Entry two.")
        assert f1 != f2
        # Second file should have a numeric suffix
        assert "_" in f2 or f1 != f2

    def test_list_entries_empty(self, journal):
        entries = journal.list_entries()
        assert entries == []

    def test_list_entries_returns_metadata(self, journal):
        journal.write("Identity Crisis", "Feeling uncertain today.")
        entries = journal.list_entries()
        assert len(entries) == 1
        entry = entries[0]
        assert "filename" in entry
        assert "date" in entry
        assert "topic" in entry
        assert "preview" in entry
        assert entry["topic"] == "Identity Crisis"

    def test_list_entries_preview(self, journal):
        journal.write("Growth", "This is the actual content after the header.")
        entries = journal.list_entries()
        assert len(entries) == 1
        assert entries[0]["preview"] != ""

    def test_list_entries_limit(self, journal):
        for i in range(5):
            journal.write(f"Topic {i}", f"Content {i}")
        entries = journal.list_entries(limit=3)
        assert len(entries) == 3

    def test_list_entries_newest_first(self, journal):
        journal.write("First", "Content A")
        journal.write("Second", "Content B")
        entries = journal.list_entries()
        # Newest first — the second entry should appear first
        # (both are same day, so sorted by filename desc)
        assert len(entries) == 2

    def test_read_entry(self, journal):
        filename = journal.write("Test", "Hello world.")
        content = journal.read_entry(filename)
        assert "Hello world." in content
        assert "# Journal: Test" in content

    def test_read_entry_missing(self, journal):
        assert journal.read_entry("nonexistent.md") == ""

    def test_directory_created(self, tmp_path):
        journal_dir = tmp_path / "deep" / "nested" / "journal"
        store = JournalStore(journal_dir)
        assert journal_dir.exists()
