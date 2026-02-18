"""Shared fixtures for Engram tests."""

import pytest
import tempfile
from pathlib import Path

from engram.core.config import Config
from engram.episodic.store import EpisodicStore
from engram.semantic.store import SemanticStore
from engram.semantic.identity import IdentityResolver
from engram.procedural.store import ProceduralStore


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory that persists for the test."""
    return tmp_path


@pytest.fixture
def config(tmp_dir):
    """Provide a Config pointing at a temp directory."""
    cfg = Config.from_data_dir(tmp_dir, signal_mode="regex", extract_mode="off")
    cfg.ensure_directories()
    return cfg


@pytest.fixture
def config_with_soul(config):
    """Config with a SOUL.md, identities.yaml, and trust.yaml pre-populated.

    Sets ``core_person="tester"`` so the trust gate is active, and
    pre-registers alice as a friend and bob as an acquaintance in both
    identity and trust files.
    """
    config.core_person = "tester"
    config.soul_path.write_text(
        "# Test Identity\n\nI am a test agent. I value honesty and curiosity.\n\n"
        "## Core Values\n- Honesty\n- Curiosity\n- Directness\n",
        encoding="utf-8",
    )
    config.identities_path.write_text(
        "people:\n"
        "  alice:\n"
        "    aliases: [alice_dev, Alice, alicew]\n"
        "    trust_tier: friend\n"
        "  bob:\n"
        "    aliases: [bobby, Robert]\n"
        "    trust_tier: acquaintance\n",
        encoding="utf-8",
    )
    # Pre-populate trust.yaml so TrustGate resolves tiers correctly.
    # Without this, SemanticStore.check_trust() returns "stranger" for everyone.
    # Note: SemanticStore expects data nested under a "tiers" key.
    trust_path = config.semantic_dir / "trust.yaml"
    trust_path.parent.mkdir(parents=True, exist_ok=True)
    trust_path.write_text(
        "tiers:\n"
        "  alice:\n"
        "    tier: friend\n"
        "    reason: Test fixture\n"
        "  bob:\n"
        "    tier: acquaintance\n"
        "    reason: Test fixture\n"
        "  tester:\n"
        "    tier: core\n"
        "    reason: Owner — registered at initialization\n",
        encoding="utf-8",
    )
    return config


@pytest.fixture
def episodic(config):
    """Provide a fresh EpisodicStore."""
    store = EpisodicStore(config.db_path)
    yield store
    store.close()


@pytest.fixture
def semantic(config):
    """Provide a fresh SemanticStore."""
    return SemanticStore(
        semantic_dir=config.semantic_dir,
        soul_dir=config.soul_dir,
    )


@pytest.fixture
def identity(config_with_soul):
    """Provide an IdentityResolver with test data."""
    return IdentityResolver(config_with_soul.identities_path)


@pytest.fixture
def procedural(config):
    """Provide a fresh ProceduralStore."""
    return ProceduralStore(config.procedural_dir)
