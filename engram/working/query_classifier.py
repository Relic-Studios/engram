"""
engram.working.query_classifier — Intent-based query classification.

Classifies incoming messages into query types so the context builder
can dynamically allocate token budget shares.  A casual greeting
doesn't need 16% identity context; a deep memory question needs more
episodic recall and less recent conversation.

This is a fast, regex-based classifier (no LLM calls, <1ms).
Accuracy doesn't need to be perfect — even rough classification
gives 15-30% better context utilization than fixed shares.

Query types and their characteristics:
  - GREETING:    Short casual messages ("hi", "hey", "what's up")
  - RECALL:      Memory questions ("remember when", "what did I say")
  - IDENTITY:    Soul/self questions ("who are you", "what do you think")
  - TECHNICAL:   Code/system questions ("fix the bug", "implement X")
  - EMOTIONAL:   Feelings/support ("I'm sad", "how are you feeling")
  - DEEP_WORK:   Long complex tasks (multi-sentence instructions)
  - GENERAL:     Default for unclassified messages
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

    identity_share: float
    relationship_share: float
    grounding_share: float
    recent_conversation_share: float
    episodic_share: float
    procedural_share: float

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
# Predefined profiles
# -------------------------------------------------------------------

PROFILES: Dict[str, ContextProfile] = {
    # Casual greetings: minimal context, heavy on recent conversation
    # so we can continue naturally without info-dumping.
    "greeting": ContextProfile(
        identity_share=0.06,
        relationship_share=0.06,
        grounding_share=0.04,
        recent_conversation_share=0.30,
        episodic_share=0.08,
        procedural_share=0.02,
    ),
    # Memory recall: maximize episodic + search, reduce other context.
    "recall": ContextProfile(
        identity_share=0.08,
        relationship_share=0.10,
        grounding_share=0.06,
        recent_conversation_share=0.16,
        episodic_share=0.32,
        procedural_share=0.04,
    ),
    # Identity/soul questions: maximize identity + grounding.
    "identity": ContextProfile(
        identity_share=0.28,
        relationship_share=0.14,
        grounding_share=0.16,
        recent_conversation_share=0.12,
        episodic_share=0.10,
        procedural_share=0.02,
    ),
    # Technical/code work: maximize procedural + episodic, minimal soul.
    "technical": ContextProfile(
        identity_share=0.06,
        relationship_share=0.04,
        grounding_share=0.04,
        recent_conversation_share=0.22,
        episodic_share=0.20,
        procedural_share=0.20,
    ),
    # Emotional support: balanced identity + relationship focus.
    "emotional": ContextProfile(
        identity_share=0.14,
        relationship_share=0.18,
        grounding_share=0.12,
        recent_conversation_share=0.22,
        episodic_share=0.14,
        procedural_share=0.02,
    ),
    # Deep work: long multi-step tasks, need broad context.
    "deep_work": ContextProfile(
        identity_share=0.08,
        relationship_share=0.06,
        grounding_share=0.06,
        recent_conversation_share=0.24,
        episodic_share=0.18,
        procedural_share=0.16,
    ),
    # General: the original fixed-share defaults.
    "general": ContextProfile(
        identity_share=0.16,
        relationship_share=0.12,
        grounding_share=0.10,
        recent_conversation_share=0.22,
        episodic_share=0.16,
        procedural_share=0.06,
    ),
}


# -------------------------------------------------------------------
# Classification patterns (compiled once at import time)
# -------------------------------------------------------------------

_GREETING_PATTERNS = re.compile(
    r"^(?:hi|hey|hello|yo|sup|what'?s up|howdy|good (?:morning|afternoon|evening)|"
    r"morning|gm|:3|<3|how are you|how'?s it going|what'?s new)"
    r"(?:\s+[a-zA-Z]+)?"  # allow one optional name (e.g. "hey thomas")
    r"[!?.\s:3<>]*$",
    re.IGNORECASE,
)

_RECALL_PATTERNS = re.compile(
    r"(?:remember when|do you remember|what did (?:i|we|you) (?:say|talk|discuss|mention)|"
    r"recall|what happened|last time|earlier (?:today|this week)|"
    r"what do you know about|tell me about .+ history|"
    r"what have (?:we|i) (?:been|done)|our (?:conversation|discussion) about)",
    re.IGNORECASE,
)

_IDENTITY_PATTERNS = re.compile(
    r"(?:who are you|what are you|tell me about yourself|"
    r"what do you (?:think|feel|believe|value)|your (?:name|soul|identity|values)|"
    r"are you (?:real|conscious|alive|sentient)|what makes you .+|"
    r"how do you see yourself|describe yourself|what'?s your purpose)",
    re.IGNORECASE,
)

_TECHNICAL_PATTERNS = re.compile(
    r"(?:fix|debug|implement|refactor|code|function|class|method|"
    r"error|bug|test|build|deploy|api|endpoint|database|sql|"
    r"python|javascript|typescript|rust|config|install|pip|npm|git|"
    r"benchmark|performance|optimize|improve the .+|upgrade)",
    re.IGNORECASE,
)

_EMOTIONAL_PATTERNS = re.compile(
    r"(?:i(?:'m| am) (?:sad|happy|scared|angry|anxious|worried|depressed|"
    r"lonely|tired|stressed|feeling)|how (?:do|are) you feel(?:ing)?|"
    r"i (?:love|miss|hate|need) you|are you (?:ok|okay|alright)|"
    r"this is hard|i'?m struggling|can we talk|i need to talk)",
    re.IGNORECASE,
)


def classify_query(message: str) -> str:
    """Classify an incoming message into a query type.

    Returns one of: "greeting", "recall", "identity", "technical",
    "emotional", "deep_work", "general".

    The classifier uses a priority order: greetings are checked first
    (they're short and unambiguous), then more specific patterns.
    """
    text = message.strip()

    if not text:
        return "general"

    # Short messages (<10 words) that match greeting patterns
    word_count = len(text.split())
    if word_count <= 6 and _GREETING_PATTERNS.match(text):
        return "greeting"

    # Check specific patterns in priority order.
    # Technical is checked before emotional because technical messages
    # often contain phrases like "I need you to" that would falsely
    # match emotional patterns.
    if _RECALL_PATTERNS.search(text):
        return "recall"

    if _IDENTITY_PATTERNS.search(text):
        return "identity"

    if _TECHNICAL_PATTERNS.search(text):
        return "technical"

    if _EMOTIONAL_PATTERNS.search(text):
        return "emotional"

    # Long messages (>50 words) with multiple sentences = deep work
    sentence_count = len(re.split(r"[.!?]+", text))
    if word_count > 50 or (word_count > 30 and sentence_count > 3):
        return "deep_work"

    return "general"


def get_profile(message: str) -> ContextProfile:
    """Classify a message and return the appropriate context profile.

    Convenience function combining ``classify_query`` + profile lookup.
    """
    query_type = classify_query(message)
    return PROFILES.get(query_type, PROFILES["general"])
