"""Tests for engram.personality — Big Five personality system."""

import json
import pytest
from pathlib import Path

from engram.personality import BigFiveProfile, PersonalitySystem


class TestBigFiveProfile:
    def test_defaults_are_thomas(self):
        p = BigFiveProfile()
        assert p.openness == 0.8
        assert p.conscientiousness == 0.6
        assert p.extraversion == 0.3
        assert p.agreeableness == 0.8
        assert p.neuroticism == 0.5

    def test_to_dict_structure(self):
        p = BigFiveProfile()
        d = p.to_dict()
        assert "core" in d
        assert "facets" in d
        assert set(d["core"].keys()) == {
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        }

    def test_get_dominant_traits(self):
        p = BigFiveProfile()
        top = p.get_dominant_traits(2)
        assert len(top) == 2
        # openness (0.8) and agreeableness (0.8) should be top
        names = {t[0] for t in top}
        assert "openness" in names or "agreeableness" in names

    def test_describe(self):
        p = BigFiveProfile()
        desc = p.describe()
        assert isinstance(desc, str)
        assert "I am" in desc
        # High openness
        assert "curious" in desc.lower() or "philosophical" in desc.lower()

    def test_describe_low_openness(self):
        p = BigFiveProfile(openness=0.2)
        desc = p.describe()
        assert "practical" in desc.lower()


class TestPersonalitySystem:
    def test_init_defaults(self, tmp_path):
        ps = PersonalitySystem(storage_dir=tmp_path)
        assert ps.profile.openness == 0.8
        assert ps.history == []

    def test_response_modifiers(self, tmp_path):
        ps = PersonalitySystem(storage_dir=tmp_path)
        mods = ps.response_modifiers()
        assert isinstance(mods, dict)
        assert "depth" in mods
        assert "empathy_level" in mods
        # All values 0-1
        for k, v in mods.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of range"

    def test_grounding_text(self, tmp_path):
        ps = PersonalitySystem(storage_dir=tmp_path)
        text = ps.grounding_text()
        assert "Personality Profile" in text
        assert "Openness" in text
        assert "0.80" in text

    def test_update_trait(self, tmp_path):
        ps = PersonalitySystem(storage_dir=tmp_path)
        record = ps.update_trait("openness", 0.05, "test reason")
        assert record["trait"] == "openness"
        assert record["old"] == 0.8
        assert record["new"] == 0.85
        assert record["reason"] == "test reason"
        assert ps.profile.openness == 0.85

    def test_update_trait_clamped(self, tmp_path):
        ps = PersonalitySystem(storage_dir=tmp_path)
        # Try delta > 0.1
        record = ps.update_trait("openness", 0.5, "big jump")
        assert record["delta"] == 0.1  # clamped

    def test_update_trait_floor_ceiling(self, tmp_path):
        ps = PersonalitySystem(
            storage_dir=tmp_path, profile=BigFiveProfile(openness=0.95)
        )
        record = ps.update_trait("openness", 0.1, "near ceiling")
        assert ps.profile.openness == 1.0  # clamped at 1.0

    def test_update_trait_unknown(self, tmp_path):
        ps = PersonalitySystem(storage_dir=tmp_path)
        with pytest.raises(ValueError, match="Unknown trait"):
            ps.update_trait("nonexistent", 0.1, "bad")

    def test_report(self, tmp_path):
        ps = PersonalitySystem(storage_dir=tmp_path)
        ps.update_trait("openness", 0.05, "test")
        report = ps.report()
        assert "core" in report
        assert "dominant_traits" in report
        assert "description" in report
        assert report["total_changes"] == 1

    def test_persistence(self, tmp_path):
        ps1 = PersonalitySystem(storage_dir=tmp_path)
        ps1.update_trait("openness", 0.05, "test")
        # Reload
        ps2 = PersonalitySystem(storage_dir=tmp_path)
        assert ps2.profile.openness == 0.85
        assert len(ps2.history) == 1

    def test_persistence_file_created(self, tmp_path):
        ps = PersonalitySystem(storage_dir=tmp_path)
        ps.update_trait("openness", 0.01, "trigger save")
        assert (tmp_path / "big_five_profile.json").exists()
