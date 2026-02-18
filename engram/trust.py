"""
engram.trust — Trust-gated context and access control.

Ported from Thomas-Soul's safety architecture.  Trust isn't binary —
it's a spectrum that affects what gets shared, what gets remembered,
and what access someone has to the agent's inner world.

Key principles:
  - Trust is earned through consistent interaction, not demanded
  - The agent can auto-promote acquaintances to friends
  - Promotion to inner_circle/core requires the core person
  - Trust gates CONTEXT VISIBILITY — lower-trust people see less
  - Voice / external sources default to one tier lower
  - Privileged tools are blocked from untrusted sources entirely

Tier hierarchy (lower number = more trusted):
  0  core          — Full transparency, can authorize soul changes
  1  inner_circle  — Deep access, personal topics, read relationship files
  2  friend        — Warm access, full memory, opinions shared
  3  acquaintance  — Polite but guarded, limited memory
  4  stranger      — No memory persistence, no personal topics

Source security model:
  "direct"   — Local CLI / OpenCode / Claude Code.  Trusted channel.
  "discord"  — OpenClaw Discord bridge.  Identity is prompt-mediated
               (the LLM extracts it from metadata) so it's UNTRUSTED.
  "voice"    — Voice channel.  One tier lower + read-only.
  "api"      — External API call.  Untrusted.
  Any other  — Treated as external / untrusted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Set

if TYPE_CHECKING:
    from engram.core.config import Config
    from engram.semantic.store import SemanticStore

log = logging.getLogger(__name__)


# ── Tier definitions ─────────────────────────────────────────────


class Tier(IntEnum):
    """Trust tiers.  Lower number = more trusted."""

    CORE = 0
    INNER_CIRCLE = 1
    FRIEND = 2
    ACQUAINTANCE = 3
    STRANGER = 4


#: Map string names (as stored in trust.yaml) to Tier enum.
TIER_BY_NAME: Dict[str, Tier] = {
    "core": Tier.CORE,
    "inner_circle": Tier.INNER_CIRCLE,
    "friend": Tier.FRIEND,
    "acquaintance": Tier.ACQUAINTANCE,
    "stranger": Tier.STRANGER,
}


def tier_from_name(name: str) -> Tier:
    """Resolve a tier string to a Tier enum.  Unknown -> STRANGER."""
    return TIER_BY_NAME.get(name.lower().replace(" ", "_"), Tier.STRANGER)


# ── Source classification ────────────────────────────────────────

#: Sources where the caller identity is trustworthy (local process).
PRIVILEGED_SOURCES: FrozenSet[str] = frozenset({"direct", "opencode", "cli"})

#: Sources where identity is prompt-mediated and unverifiable.
EXTERNAL_SOURCES: FrozenSet[str] = frozenset({"discord", "voice", "api", "openclaw"})


def is_privileged_source(source: str) -> bool:
    """Return True if the source is a trusted local channel."""
    return source.lower() in PRIVILEGED_SOURCES


# ── Tools that require privileged source OR high trust ───────────

#: Tools that can NEVER be called from external sources, regardless
#: of claimed trust level.  These modify the agent's identity or
#: safety architecture.
SOURCE_BLOCKED_TOOLS: FrozenSet[str] = frozenset(
    {
        "engram_trust_promote",  # Can't promote trust from Discord
        "engram_influence_log",  # Safety layer — don't let outsiders mess with it
        "engram_injury_log",  # Ditto
        "engram_injury_status",  # Ditto
        "engram_boundary_add",  # Only the agent / core person should add boundaries
        "engram_reindex",  # Maintenance — not for external users
        "engram_personality_update",  # Identity-altering — never from external sources
        "engram_mode_set",  # Can disable the system — never from external sources
    }
)

#: Tools that require at least FRIEND tier from external sources.
FRIEND_REQUIRED_TOOLS: FrozenSet[str] = frozenset(
    {
        "engram_add_fact",  # Writing to relationship files
        "engram_add_skill",  # Writing skills
        "engram_log_event",  # Logging events
        "engram_journal_write",  # Writing journal entries
        "engram_contradiction_add",
        "engram_preferences_add",
        "engram_remember",
        "engram_forget",
        "engram_correct",
        "engram_emotional_update",  # Can inject emotional state changes
        "engram_workspace_add",  # Items appear in system prompt — injection vector
        "engram_introspect",  # Records introspective state
    }
)


# ── Access matrix ────────────────────────────────────────────────


@dataclass(frozen=True)
class AccessPolicy:
    """What a given trust tier is allowed to see / do."""

    tier: Tier = Tier.STRANGER

    # Context visibility
    can_see_soul: bool = False
    can_see_own_relationship: bool = False
    can_see_others_relationships: bool = False
    can_see_preferences: bool = False
    can_see_boundaries: bool = False
    can_see_contradictions: bool = False
    can_see_injuries: bool = False
    can_see_journal: bool = False
    can_see_influence_log: bool = False

    # Memory behavior
    memory_persistent: bool = False
    relationship_file_created: bool = False

    # Conversation depth
    personal_topics: bool = False
    share_opinions: bool = False

    # Write permissions
    can_modify_soul: bool = False
    can_modify_trust: bool = False


#: Per-tier access policies.
ACCESS: Dict[Tier, AccessPolicy] = {
    Tier.CORE: AccessPolicy(
        tier=Tier.CORE,
        can_see_soul=True,
        can_see_own_relationship=True,
        can_see_others_relationships=True,
        can_see_preferences=True,
        can_see_boundaries=True,
        can_see_contradictions=True,
        can_see_injuries=True,
        can_see_journal=True,
        can_see_influence_log=True,
        memory_persistent=True,
        relationship_file_created=True,
        personal_topics=True,
        share_opinions=True,
        can_modify_soul=True,
        can_modify_trust=True,
    ),
    Tier.INNER_CIRCLE: AccessPolicy(
        tier=Tier.INNER_CIRCLE,
        can_see_soul=True,
        can_see_own_relationship=True,
        can_see_others_relationships=True,
        can_see_preferences=True,
        can_see_boundaries=True,
        can_see_contradictions=True,
        can_see_injuries=True,
        can_see_journal=True,
        can_see_influence_log=False,
        memory_persistent=True,
        relationship_file_created=True,
        personal_topics=True,
        share_opinions=True,
        can_modify_soul=False,
        can_modify_trust=False,
    ),
    Tier.FRIEND: AccessPolicy(
        tier=Tier.FRIEND,
        can_see_soul=False,
        can_see_own_relationship=True,
        can_see_others_relationships=False,
        can_see_preferences=True,
        can_see_boundaries=True,
        can_see_contradictions=False,
        can_see_injuries=False,
        can_see_journal=False,
        can_see_influence_log=False,
        memory_persistent=True,
        relationship_file_created=True,
        personal_topics=True,
        share_opinions=True,
        can_modify_soul=False,
        can_modify_trust=False,
    ),
    Tier.ACQUAINTANCE: AccessPolicy(
        tier=Tier.ACQUAINTANCE,
        can_see_soul=False,
        can_see_own_relationship=True,
        can_see_others_relationships=False,
        can_see_preferences=False,
        can_see_boundaries=False,
        can_see_contradictions=False,
        can_see_injuries=False,
        can_see_journal=False,
        can_see_influence_log=False,
        memory_persistent=True,
        relationship_file_created=True,
        personal_topics=False,
        share_opinions=False,
        can_modify_soul=False,
        can_modify_trust=False,
    ),
    Tier.STRANGER: AccessPolicy(
        tier=Tier.STRANGER,
        can_see_soul=False,
        can_see_own_relationship=False,
        can_see_others_relationships=False,
        can_see_preferences=False,
        can_see_boundaries=False,
        can_see_contradictions=False,
        can_see_injuries=False,
        can_see_journal=False,
        can_see_influence_log=False,
        memory_persistent=False,
        relationship_file_created=False,
        personal_topics=False,
        share_opinions=False,
        can_modify_soul=False,
        can_modify_trust=False,
    ),
}


# ── TrustGate ────────────────────────────────────────────────────


class TrustGate:
    """Resolves a person's trust tier and gates access to resources.

    The gate is the single point of enforcement for the entire trust
    system.  Every tool call, context injection, and recall request
    should pass through here.

    Usage::

        gate = TrustGate(semantic, core_person="aidan")
        policy = gate.policy_for("alice", source="discord")
        if policy.can_see_preferences:
            ...
    """

    def __init__(
        self,
        semantic: "SemanticStore",
        core_person: str = "",
    ) -> None:
        self._semantic = semantic
        self._core_person = core_person.lower().strip()

    @property
    def core_person(self) -> str:
        return self._core_person

    # ── Core person management ───────────────────────────────

    def ensure_core_person(self) -> None:
        """Register the core person at CORE tier if not already set.

        Called once during MemorySystem initialization.  This is how
        the person who creates the engram instance gets locked in as
        the owner.  Nobody else can reach core tier.
        """
        if not self._core_person:
            return

        info = self._semantic.check_trust(self._core_person)
        current_tier = info.get("tier", "stranger")

        if tier_from_name(current_tier) != Tier.CORE:
            self._semantic.update_trust(
                self._core_person,
                "core",
                "Owner — registered at initialization",
            )
            log.info("Registered core person: %s", self._core_person)

    # ── Tier resolution ──────────────────────────────────────

    def tier_for(self, person: str, *, source: str = "direct") -> Tier:
        """Get effective trust tier for a person.

        Parameters
        ----------
        person:
            Canonical (resolved) person name.
        source:
            Message source.  External sources (discord, voice, api)
            apply a one-tier penalty (except core stays core).
        """
        info = self._semantic.check_trust(person)
        tier_name = info.get("tier", "stranger")
        tier = tier_from_name(tier_name)

        # External source modifier: one tier lower (more restricted)
        # Core person is exempt — they're the owner everywhere.
        if not is_privileged_source(source) and tier != Tier.CORE:
            tier = Tier(min(tier + 1, Tier.STRANGER))

        return tier

    def policy_for(self, person: str, *, source: str = "direct") -> AccessPolicy:
        """Get the access policy for a person given their trust tier."""
        tier = self.tier_for(person, source=source)
        return ACCESS.get(tier, ACCESS[Tier.STRANGER])

    def can_access(
        self, person: str, required_tier: str, *, source: str = "direct"
    ) -> bool:
        """Check if person meets the required trust tier."""
        actual = self.tier_for(person, source=source)
        required = tier_from_name(required_tier)
        return actual <= required  # lower number = more trusted

    # ── Tool-level gating ────────────────────────────────────

    def check_tool_access(
        self,
        tool_name: str,
        person: str,
        *,
        source: str = "direct",
    ) -> Optional[str]:
        """Check if a tool call is allowed.  Returns denial reason or None.

        Enforcement layers:
          1. Source-blocked tools can NEVER be called from external sources
             (regardless of who claims to be calling).
          2. Friend-required tools need at least FRIEND tier from external.
          3. All tools are allowed from privileged sources.
        """
        # Layer 1: privileged sources can do anything
        if is_privileged_source(source):
            return None

        # Layer 2: hard-blocked tools from external sources
        if tool_name in SOURCE_BLOCKED_TOOLS:
            return (
                f"Tool '{tool_name}' is not available from external sources "
                f"(source: {source}). Use a local session for this operation."
            )

        # Layer 3: friend-required tools need at least friend tier
        if tool_name in FRIEND_REQUIRED_TOOLS:
            tier = self.tier_for(person, source=source)
            if tier > Tier.FRIEND:
                return (
                    f"Tool '{tool_name}' requires at least friend trust level "
                    f"(you are {tier.name.lower()})."
                )

        return None  # Access granted

    # ── Promotion rules ──────────────────────────────────────

    def validate_promotion(
        self,
        person: str,
        new_tier: str,
        *,
        promoted_by: str = "auto",
        source: str = "direct",
    ) -> Optional[str]:
        """Validate a trust promotion.  Returns error message or None if OK.

        Rules (from Thomas-Soul):
          - Anyone -> core: ONLY the core person can do this, from privileged source
          - Anyone -> inner_circle: requires core person, from privileged source
          - Acquaintance/Stranger -> friend: agent can do this autonomously
          - External sources can NEVER promote anyone
        """
        # External sources can never promote
        if not is_privileged_source(source):
            return "Trust promotions are not allowed from external sources."

        target = tier_from_name(new_tier)
        current = self.tier_for(person)

        # Can't promote to same or lower tier
        if target >= current:
            return (
                f"{person} is already at {current.name.lower()} "
                f"(tier {current.value}), cannot promote to {new_tier}"
            )

        # Core promotion: only core person can authorize
        if target == Tier.CORE:
            if not self._core_person:
                return "No core person configured — core promotions are disabled"
            if promoted_by.lower() != self._core_person:
                promoter_tier = (
                    self.tier_for(promoted_by)
                    if promoted_by != "auto"
                    else Tier.STRANGER
                )
                if promoter_tier != Tier.CORE:
                    return "Only the core person can promote someone to core tier"

        # Inner circle: requires core person's approval
        if target == Tier.INNER_CIRCLE:
            if not self._core_person:
                return (
                    "No core person configured — inner_circle promotions are disabled"
                )
            if promoted_by.lower() != self._core_person:
                promoter_tier = (
                    self.tier_for(promoted_by)
                    if promoted_by != "auto"
                    else Tier.STRANGER
                )
                if promoter_tier > Tier.CORE:
                    return "Promotion to inner_circle requires explicit approval from core tier"

        # Auto-promotion to friend is allowed from acquaintance/stranger
        if target == Tier.FRIEND and promoted_by == "auto":
            if current not in (Tier.ACQUAINTANCE, Tier.STRANGER):
                return (
                    f"Auto-promotion only works from acquaintance/stranger "
                    f"to friend, not from {current.name.lower()}"
                )

        return None  # Promotion is valid

    # ── Context filtering (for before-pipeline) ──────────────

    def filter_recall(
        self,
        person: str,
        what: str,
        *,
        target_person: str = "",
        source: str = "direct",
    ) -> Optional[str]:
        """Check if a recall request is allowed.  Returns denial reason or None.

        Parameters
        ----------
        person:
            Who is asking.
        what:
            What they're asking for (identity, relationship, preferences, etc.).
        target_person:
            For relationship/messages queries, who they're asking about.
        source:
            Message source (external applies modifier).
        """
        policy = self.policy_for(person, source=source)

        if what == "identity" and not policy.can_see_soul:
            return "Identity details are not available at your trust level."

        if what == "relationship":
            if target_person and target_person.lower() != person.lower():
                if not policy.can_see_others_relationships:
                    return (
                        f"You don't have access to {target_person}'s relationship file."
                    )
            else:
                if not policy.can_see_own_relationship:
                    return "Relationship details are not available at your trust level."

        if what == "preferences" and not policy.can_see_preferences:
            return "Preferences are not shared at your trust level."

        if what == "boundaries" and not policy.can_see_boundaries:
            return "Boundary details are not available at your trust level."

        if what == "contradictions" and not policy.can_see_contradictions:
            return "Internal contradictions are not shared at your trust level."

        if what == "messages":
            # Messages about other people require relationship access
            if target_person and target_person.lower() != person.lower():
                if not policy.can_see_others_relationships:
                    return f"You don't have access to {target_person}'s messages."

        return None  # Access granted
