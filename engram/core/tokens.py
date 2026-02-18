"""
engram.core.tokens — Token estimation utilities.

Kept in its own module so callers that only need token math
don't have to import the full types module.

Uses tiktoken when available for accurate counts, otherwise
falls back to a regex-based heuristic that models BPE token
boundaries (~10% error vs ~40% for the naive ``len//4`` approach).
"""

from __future__ import annotations

import re

# Try to load tiktoken for accurate token counting.  It's an optional
# dependency — the system degrades gracefully to heuristic mode.
_tiktoken_encoding = None
try:
    import tiktoken as _tiktoken

    _tiktoken_encoding = _tiktoken.get_encoding("cl100k_base")
except (ImportError, Exception):
    pass

# Regex that approximates BPE token boundaries:
#   - words (including contractions like don't)
#   - runs of digits
#   - individual punctuation / special characters
#   - whitespace runs (BPE typically merges leading spaces with words)
_TOKEN_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d"""  # contractions
    r"""|[a-zA-Z]+"""  # words
    r"""|[0-9]+"""  # digit runs
    r"""|\S""",  # individual non-space characters
    re.VERBOSE,
)


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text*.

    When tiktoken is installed, uses ``cl100k_base`` encoding for
    exact counts.  Otherwise falls back to a regex-based heuristic
    that approximates BPE boundaries (typically within ~10% of the
    true count for English prose).

    Always returns at least 1.
    """
    if not text:
        return 1
    if _tiktoken_encoding is not None:
        return max(1, len(_tiktoken_encoding.encode(text)))
    # Heuristic: count regex-matched token-like segments
    return max(1, len(_TOKEN_PATTERN.findall(text)))


def estimate_tokens_messages(messages: list[dict]) -> int:
    """Estimate total tokens across a list of message dicts.

    Each message is expected to have at least a ``"content"`` key.
    Adds a small per-message overhead (4 tokens) to account for
    role / name / separator tokens.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += estimate_tokens(content) + 4  # per-message overhead
    return total


def fits_budget(text: str, budget: int) -> bool:
    """Return True if *text* fits within *budget* tokens."""
    return estimate_tokens(text) <= budget


def trim_to_budget(text: str, budget: int, suffix: str = "...") -> str:
    """Trim *text* so its estimated token count fits *budget*.

    Trims at character boundaries (not token boundaries) and appends
    *suffix* to indicate truncation.  Returns the original text
    unchanged if it already fits.
    """
    if fits_budget(text, budget):
        return text

    # target char count: budget * 4 minus room for suffix
    target_chars = max(0, budget * 4 - len(suffix))
    return text[:target_chars] + suffix
