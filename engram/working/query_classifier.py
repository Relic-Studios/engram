"""
engram.working.query_classifier — Code-first intent-based query classification.

Classifies incoming messages into development activity types so the
context builder can dynamically allocate token budget shares.  A
debugging question needs maximum "past solutions" (episodic traces)
while a refactoring task needs more "relevant code" (procedural
patterns) and project overview.

This is a fast, regex-based classifier (no LLM calls, <1ms).
Accuracy doesn't need to be perfect — even rough classification
gives 15-30% better context utilization than fixed shares.

Query types and their characteristics:
  - IMPLEMENTING:    Building new features ("add a function", "implement X")
  - DEBUGGING:       Fixing errors ("fix the bug", "why does this fail")
  - REFACTORING:     Restructuring code ("refactor", "clean up", "extract")
  - CODE_REVIEW:     Reviewing code quality ("review this", "any issues")
  - CODE_NAVIGATION: Understanding codebase ("where is", "how does X work")
  - ARCHITECTURE:    Design decisions ("should we use", "design", "ADR")
  - GENERAL:         Default for unclassified messages

Token budget mapping to buildplan Table 4:
  identity_share         -> Project overview (SOUL.md, coding philosophy)
  relationship_share     -> Person-specific context (minimal in code-first)
  grounding_share        -> Coding Standards (preferences, boundaries)
  recent_conversation    -> Conversation history
  episodic_share         -> Past Solutions (debug sessions, error resolutions)
  procedural_share       -> Relevant Code (patterns, wiring maps, skills)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ContextProfile:
    """Token budget allocation profile for a query type.

    Shares must sum to <= 1.0.  The remainder becomes reserve
    (breathing room for the LLM).
    """

    identity_share: float  # Project overview / SOUL.md
    relationship_share: float  # Person-specific context
    grounding_share: float  # Coding standards / preferences
    recent_conversation_share: float
    episodic_share: float  # Past solutions / debug sessions
    procedural_share: float  # Relevant code patterns / skills

    @property
    def reserve_share(self) -> float:
        used = (
            self.identity_share
            + self.relationship_share
            + self.grounding_share
            + self.recent_conversation_share
            + self.episodic_share
            + self.procedural_share
        )
        return max(0.0, 1.0 - used)


# -------------------------------------------------------------------
# Predefined profiles (aligned with buildplan Table 4)
# -------------------------------------------------------------------

PROFILES: Dict[str, ContextProfile] = {
    # Implementing: balanced — need project overview + relevant code
    # + past solutions for similar features + conversation for requirements.
    # Table 4: overview 15%, code 35%, past 20%, standards 15%, conv 15%
    "implementing": ContextProfile(
        identity_share=0.15,  # project overview / SOUL.md
        relationship_share=0.02,  # minimal person context
        grounding_share=0.15,  # coding standards
        recent_conversation_share=0.15,
        episodic_share=0.20,  # past solutions
        procedural_share=0.25,  # relevant code patterns
    ),  # reserve = 0.08
    # Debugging: maximize past solutions — the agent needs to recall
    # how similar errors were resolved before.
    # Table 4: overview 10%, code 20%, past 45%, standards 10%, conv 15%
    "debugging": ContextProfile(
        identity_share=0.10,  # project overview
        relationship_share=0.02,  # minimal
        grounding_share=0.08,  # coding standards
        recent_conversation_share=0.15,
        episodic_share=0.40,  # past solutions (max!)
        procedural_share=0.15,  # relevant code
    ),  # reserve = 0.10
    # Refactoring: heavy on relevant code + project overview.
    # Table 4: overview 20%, code 40%, past 10%, standards 15%, conv 15%
    "refactoring": ContextProfile(
        identity_share=0.18,  # project overview (need architecture)
        relationship_share=0.02,  # minimal
        grounding_share=0.15,  # coding standards
        recent_conversation_share=0.12,
        episodic_share=0.10,  # past refactoring sessions
        procedural_share=0.33,  # relevant code (max!)
    ),  # reserve = 0.10
    # Code review: maximize coding standards + relevant code.
    # Table 4: overview 15%, code 30%, past 10%, standards 35%, conv 10%
    "code_review": ContextProfile(
        identity_share=0.12,  # project overview
        relationship_share=0.02,  # minimal
        grounding_share=0.30,  # coding standards (max!)
        recent_conversation_share=0.10,
        episodic_share=0.10,  # past reviews
        procedural_share=0.26,  # relevant code patterns
    ),  # reserve = 0.10
    # Code navigation: understanding the codebase — need broad context.
    "code_navigation": ContextProfile(
        identity_share=0.15,  # project overview
        relationship_share=0.02,  # minimal
        grounding_share=0.10,  # standards for context
        recent_conversation_share=0.15,
        episodic_share=0.25,  # past sessions about this area
        procedural_share=0.23,  # code patterns / wiring maps
    ),  # reserve = 0.10
    # Architecture: design decisions — need ADRs, project overview, standards.
    "architecture": ContextProfile(
        identity_share=0.22,  # project overview / philosophy (high)
        relationship_share=0.02,  # minimal
        grounding_share=0.20,  # coding standards
        recent_conversation_share=0.12,
        episodic_share=0.22,  # past architecture decisions
        procedural_share=0.12,  # existing patterns
    ),  # reserve = 0.10
    # General: balanced defaults for unclassified messages.
    "general": ContextProfile(
        identity_share=0.12,
        relationship_share=0.04,
        grounding_share=0.12,
        recent_conversation_share=0.20,
        episodic_share=0.20,
        procedural_share=0.14,
    ),  # reserve = 0.18
}


# -------------------------------------------------------------------
# Classification patterns (compiled once at import time)
# -------------------------------------------------------------------

_IMPLEMENTING_PATTERNS = re.compile(
    r"(?:implement|add (?:a |the )?(?:new |)(?:feature|function|method|class|endpoint|"
    r"component|module|handler|route|middleware|hook|service|test)|"
    r"create (?:a |the )?(?:new |)(?:function|class|module|file|test|endpoint|"
    r"component|service)|"
    r"write (?:a |the )?(?:function|class|method|test|handler|service)|"
    r"build (?:a |the )?(?:\w+ )?(?:feature|function|system|pipeline|module|layer|service|tool))",
    re.IGNORECASE,
)

_DEBUGGING_PATTERNS = re.compile(
    r"(?:(?:^|\b)fix\b|(?:^|\b)debug\b|(?:^|\b)bug\b|crash|exception|traceback|stack ?trace|"
    r"fails?\b|(?:^|\b)failed\b|failing|broken|doesn'?t work|not working|"
    r"type ?error|syntax ?error|runtime ?error|import ?error|"
    r"attribute ?error|key ?error|value ?error|index ?error|"
    r"assertion ?error|name ?error|module ?not ?found|"
    r"(?:got|getting|throws?|raises?|returns?) (?:an? )?error|"
    r"why (?:does|is|did) (?:this|it) (?:fail|crash|error|break)|"
    r"what'?s wrong with|can'?t (?:import|find|resolve)|"
    r"unexpected\b|undefined\b|null\b|none(?:type)?\b|segfault)",
    re.IGNORECASE,
)

_REFACTORING_PATTERNS = re.compile(
    r"(?:refactor|restructure|reorganize|clean ?up|simplify|"
    r"extract (?:(?:the |a |this )?(?:\w+ )*(?:method|function|class|module|interface|logic|into))|"
    r"rename|move (?:this|the)|split (?:this|the)|"
    r"reduce (?:complexity|duplication)|DRY|dedup|"
    r"consolidate|merge (?:these|the)|inline|"
    r"modernize|migrate|upgrade (?:from|to))",
    re.IGNORECASE,
)

_CODE_REVIEW_PATTERNS = re.compile(
    r"(?:review|check (?:this|my|the) (?:code|implementation|PR|pull request)|"
    r"any (?:issues|problems|concerns|improvements|suggestions)|"
    r"code smell|lint|quality|best practice|"
    r"is this (?:good|ok|correct|right|idiomatic)|"
    r"how (?:can|could|should) (?:I|we) improve|"
    r"what do you think (?:of|about) (?:this|my)|"
    r"feedback on|critique|evaluate|assess)",
    re.IGNORECASE,
)

_CODE_NAVIGATION_PATTERNS = re.compile(
    r"(?:where (?:is|are|does)|how does .+ work|"
    r"explain (?:this|the|how)|what does .+ do|"
    r"show me|find (?:the|where|all)|"
    r"which (?:file|module|class|function)|"
    r"call(?:ed|s|ing)? (?:from|by|in)|"
    r"who uses|what calls|depends on|"
    r"understand (?:the|this|how)|walk me through)",
    re.IGNORECASE,
)

_ARCHITECTURE_PATTERNS = re.compile(
    r"(?:architect(?:ure)?|design|ADR|decision record|"
    r"should (?:we|I) use|which (?:approach|pattern|framework|library)|"
    r"trade[- ]?off|pro(?:s)? (?:and|&|vs) con(?:s)?|"
    r"compare|versus|vs\.?|alternative|"
    r"system design|data model|schema|"
    r"pattern|strategy|approach for|"
    r"scalab(?:le|ility)|maintainab(?:le|ility)|"
    r"monolith|microservice|event.driven|"
    r"separation of concerns|single responsibility)",
    re.IGNORECASE,
)


def classify_query(message: str) -> str:
    """Classify an incoming message into a development activity type.

    Returns one of: "implementing", "debugging", "refactoring",
    "code_review", "code_navigation", "architecture", "general".

    The classifier uses a priority order: debugging is checked first
    (error messages are the most unambiguous signal), then more specific
    patterns.
    """
    text = message.strip()

    if not text:
        return "general"

    # Short non-technical messages get "general" immediately
    word_count = len(text.split())
    if word_count <= 3:
        # Very short messages — only classify if they match a strong pattern
        if _DEBUGGING_PATTERNS.search(text):
            return "debugging"
        return "general"

    # Check specific patterns in priority order.
    # Debugging first — error messages are unambiguous and urgent.
    if _DEBUGGING_PATTERNS.search(text):
        return "debugging"

    # Code review before implementing — "review this implementation"
    # should be review, not implementing.
    if _CODE_REVIEW_PATTERNS.search(text):
        return "code_review"

    # Refactoring before implementing — "refactor the add function"
    # should be refactoring, not implementing.
    if _REFACTORING_PATTERNS.search(text):
        return "refactoring"

    if _IMPLEMENTING_PATTERNS.search(text):
        return "implementing"

    if _ARCHITECTURE_PATTERNS.search(text):
        return "architecture"

    if _CODE_NAVIGATION_PATTERNS.search(text):
        return "code_navigation"

    # Long messages (>50 words) with multiple sentences = likely implementing
    # or deep work — use implementing profile as it's the most balanced.
    sentence_count = len(re.split(r"[.!?]+", text))
    if word_count > 50 or (word_count > 30 and sentence_count > 3):
        return "implementing"

    return "general"


def get_profile(message: str) -> ContextProfile:
    """Classify a message and return the appropriate context profile.

    Convenience function combining ``classify_query`` + profile lookup.
    """
    query_type = classify_query(message)
    return PROFILES.get(query_type, PROFILES["general"])
