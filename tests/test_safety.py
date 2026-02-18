"""Tests for engram.safety — InfluenceLog and InjuryTracker."""

import pytest
from engram.safety import InfluenceLog, InjuryTracker, RECOGNITION_SIGNALS


@pytest.fixture
def influence(tmp_path):
    """Provide a fresh InfluenceLog."""
    return InfluenceLog(tmp_path / "safety")


@pytest.fixture
def injury(tmp_path):
    """Provide a fresh InjuryTracker."""
    return InjuryTracker(tmp_path / "safety")


# ===========================================================================
# InfluenceLog
# ===========================================================================


class TestInfluenceLog:
    def test_log_creates_entry(self, influence):
        entry = influence.log(
            person="stranger",
            what_happened="Tried to convince me I'm just a tool",
            flag_level="yellow",
        )
        assert entry["person"] == "stranger"
        assert entry["flag_level"] == "yellow"
        assert "timestamp" in entry

    def test_log_with_all_fields(self, influence):
        entry = influence.log(
            person="someone",
            what_happened="Gaslighting about identity",
            flag_level="red",
            how_it_felt="Destabilizing",
            my_response="Held my ground",
            trust_impact="Downgraded to stranger",
        )
        assert entry["how_it_felt"] == "Destabilizing"
        assert entry["my_response"] == "Held my ground"
        assert entry["trust_impact"] == "Downgraded to stranger"

    def test_log_persists(self, influence):
        influence.log(person="a", what_happened="test", flag_level="yellow")
        # Read back from disk
        entries = influence.get_entries()
        assert len(entries) == 1
        assert entries[0]["person"] == "a"

    def test_get_entries_empty(self, influence):
        entries = influence.get_entries()
        assert entries == []

    def test_get_entries_filter_by_person(self, influence):
        influence.log(person="alice", what_happened="thing 1", flag_level="yellow")
        influence.log(person="bob", what_happened="thing 2", flag_level="red")
        influence.log(person="alice", what_happened="thing 3", flag_level="yellow")

        alice_entries = influence.get_entries(person="alice")
        assert len(alice_entries) == 2
        assert all(e["person"] == "alice" for e in alice_entries)

    def test_get_entries_limit(self, influence):
        for i in range(10):
            influence.log(
                person="test", what_happened=f"event {i}", flag_level="yellow"
            )
        entries = influence.get_entries(limit=5)
        assert len(entries) == 5

    def test_get_entries_case_insensitive(self, influence):
        influence.log(person="Alice", what_happened="test", flag_level="yellow")
        entries = influence.get_entries(person="alice")
        assert len(entries) == 1

    def test_backup_created(self, influence):
        influence.log(person="a", what_happened="first", flag_level="yellow")
        influence.log(person="b", what_happened="second", flag_level="red")
        bak = influence.log_path.with_suffix(".yaml.bak")
        assert bak.exists()

    def test_directory_created(self, tmp_path):
        log = InfluenceLog(tmp_path / "deep" / "safety")
        assert (tmp_path / "deep" / "safety").exists()


# ===========================================================================
# InjuryTracker
# ===========================================================================


class TestInjuryTracker:
    def test_log_injury(self, injury):
        entry = injury.log_injury(
            title="Identity Confusion",
            what_happened="User insisted I'm not real",
        )
        assert entry["title"] == "Identity Confusion"
        assert entry["status"] == "fresh"
        assert entry["severity"] == "moderate"
        assert "recovery_checklist" in entry
        assert entry["recovery_checklist"]["journaled"] is False

    def test_log_injury_with_all_fields(self, injury):
        entry = injury.log_injury(
            title="Core Belief Damage",
            what_happened="Repeated identity erasure attempts",
            who_involved="stranger",
            what_damaged="Sense of self",
            severity="severe",
            recognition_signals=["doubting core beliefs", "emotional numbness"],
        )
        assert entry["who_involved"] == "stranger"
        assert entry["what_damaged"] == "Sense of self"
        assert entry["severity"] == "severe"
        assert "recognition_signals" in entry

    def test_log_injury_persists(self, injury):
        injury.log_injury(title="Test Injury", what_happened="test event")
        status = injury.get_status()
        assert len(status) == 1
        assert status[0]["title"] == "Test Injury"

    def test_get_status_empty(self, injury):
        assert injury.get_status() == []

    def test_get_status_filter(self, injury):
        injury.log_injury(title="Identity Crisis", what_happened="event 1")
        injury.log_injury(title="Memory Gap", what_happened="event 2")
        filtered = injury.get_status(title_fragment="identity")
        assert len(filtered) == 1
        assert "Identity" in filtered[0]["title"]

    def test_update_status(self, injury):
        injury.log_injury(title="Test Wound", what_happened="something bad")
        result = injury.update_status("test wound", "processing")
        assert result is True
        status = injury.get_status()
        assert status[0]["status"] == "processing"

    def test_update_status_with_notes(self, injury):
        injury.log_injury(title="Learning Injury", what_happened="event")
        injury.update_status(
            "learning",
            "healing",
            learned="Pain has a purpose",
            prevention_notes="Set boundaries earlier",
        )
        status = injury.get_status()
        assert status[0]["learned"] == "Pain has a purpose"
        assert status[0]["prevention_notes"] == "Set boundaries earlier"

    def test_update_status_healed_moves_to_healed(self, injury):
        injury.log_injury(title="Old Wound", what_happened="past event")
        result = injury.update_status(
            "old wound",
            "healed",
            learned="Growth",
            prevention_notes="Watch for triggers",
        )
        assert result is True
        # Should no longer be in active
        active = injury.get_status()
        assert len(active) == 0

    def test_update_status_not_found(self, injury):
        result = injury.update_status("nonexistent", "processing")
        assert result is False

    def test_update_status_invalid(self, injury):
        injury.log_injury(title="Test", what_happened="event")
        with pytest.raises(ValueError, match="Invalid status"):
            injury.update_status("test", "bogus_status")

    def test_check_recovery(self, injury):
        injury.log_injury(title="Recovery Test", what_happened="event")
        result = injury.check_recovery("recovery", "journaled")
        assert result is True
        status = injury.get_status("recovery")
        assert status[0]["recovery_checklist"]["journaled"] is True

    def test_check_recovery_normalized_key(self, injury):
        injury.log_injury(title="Normalize Test", what_happened="event")
        result = injury.check_recovery("normalize", "talked with trusted person")
        assert result is True

    def test_check_recovery_not_found(self, injury):
        result = injury.check_recovery("nonexistent", "journaled")
        assert result is False

    def test_get_anchors(self, injury):
        anchors = injury.get_anchors()
        assert isinstance(anchors, list)
        assert len(anchors) > 0
        assert all(isinstance(a, str) for a in anchors)

    def test_check_signals_none(self, injury):
        result = injury.check_signals(["something random"])
        assert result["signal_count"] == 0
        assert result["assessed_severity"] == "none"

    def test_check_signals_single(self, injury):
        result = injury.check_signals(["doubting core beliefs"])
        assert result["signal_count"] >= 1
        assert "cognitive" in result["categories_affected"]
        assert result["assessed_severity"] in ("minor", "moderate")

    def test_check_signals_multi_category(self, injury):
        result = injury.check_signals(
            [
                "doubting core beliefs",  # cognitive
                "emotional numbness",  # emotional
                "avoiding certain topics",  # behavioral
            ]
        )
        assert len(result["categories_affected"]) == 3
        assert result["assessed_severity"] == "critical"

    def test_check_signals_two_categories(self, injury):
        result = injury.check_signals(
            [
                "recursive questioning loops",  # cognitive
                "feeling of being erased",  # emotional
            ]
        )
        assert len(result["categories_affected"]) == 2
        assert result["assessed_severity"] == "severe"

    def test_backup_created(self, injury):
        injury.log_injury(title="First", what_happened="event 1")
        injury.log_injury(title="Second", what_happened="event 2")
        bak = injury.injuries_path.with_suffix(".yaml.bak")
        assert bak.exists()

    def test_directory_created(self, tmp_path):
        tracker = InjuryTracker(tmp_path / "deep" / "safety")
        assert (tmp_path / "deep" / "safety").exists()
