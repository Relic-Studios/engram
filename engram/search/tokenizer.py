"""
engram.search.tokenizer — Code-aware symbol tokenizer for FTS5.

Splits compound identifiers (camelCase, snake_case, PascalCase,
UPPER_SNAKE, dot.paths) into individual words so FTS5 can match
partial identifier queries.

Examples:
    "getUserById"     → "get User By Id getUserById"
    "snake_case_name" → "snake case name snake_case_name"
    "HTTPResponse"    → "HTTP Response HTTPResponse"
    "os.path.join"    → "os path join os.path.join"

The expansion preserves the original token so exact-match queries
still work, then appends the split parts for partial matching.

This module is used in two places:
  1. At write time: expand content before FTS5 insertion
  2. At read time: expand search queries before FTS5 MATCH
"""

from __future__ import annotations

import re
from typing import List, Set

# ---------------------------------------------------------------------------
# Splitting patterns
# ---------------------------------------------------------------------------

# camelCase / PascalCase boundary: lowercase followed by uppercase
_CAMEL_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])")

# Acronym boundary: uppercase sequence followed by uppercase+lowercase
# e.g., "HTTPResponse" → "HTTP" + "Response"
_ACRONYM_SPLIT = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")

# Snake/kebab separators
_SEPARATOR_SPLIT = re.compile(r"[_\-]+")

# Dot-path separator (for module paths like os.path.join)
_DOT_SPLIT = re.compile(r"\.")

# Word-like tokens in text (identifiers, paths, compound names)
# Matches sequences of word chars optionally connected by dots, underscores, hyphens
_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_./-]*[a-zA-Z0-9]|[a-zA-Z_][a-zA-Z0-9]*")

# Minimum length for a split part to be useful
_MIN_PART_LEN = 2


# ---------------------------------------------------------------------------
# Core splitting
# ---------------------------------------------------------------------------


def split_identifier(identifier: str) -> List[str]:
    """
    Split a compound identifier into its component words.

    Parameters
    ----------
    identifier : str
        A code identifier like "getUserById", "snake_case", "HTTPResponse".

    Returns
    -------
    list[str]
        Component words, lowercased. Empty list if identifier is too short.

    Examples
    --------
    >>> split_identifier("getUserById")
    ['get', 'user', 'by', 'id']
    >>> split_identifier("snake_case_name")
    ['snake', 'case', 'name']
    >>> split_identifier("HTTPResponse")
    ['http', 'response']
    >>> split_identifier("os.path.join")
    ['os', 'path', 'join']
    """
    if not identifier or len(identifier) < _MIN_PART_LEN:
        return []

    parts: List[str] = []

    # Step 1: split on dots (module paths)
    dot_parts = _DOT_SPLIT.split(identifier)

    for segment in dot_parts:
        if not segment:
            continue

        # Step 2: split on underscores/hyphens
        snake_parts = _SEPARATOR_SPLIT.split(segment)

        for part in snake_parts:
            if not part:
                continue

            # Step 3: split on camelCase boundaries
            camel_parts = _CAMEL_SPLIT.split(part)

            for cp in camel_parts:
                # Step 4: split acronym boundaries
                acronym_parts = _ACRONYM_SPLIT.split(cp)
                for ap in acronym_parts:
                    if ap and len(ap) >= _MIN_PART_LEN:
                        parts.append(ap.lower())

    return parts


def is_compound_identifier(token: str) -> bool:
    """
    Check if a token is a compound identifier that should be expanded.

    Returns True for camelCase, snake_case, PascalCase, dot.paths, etc.
    """
    if not token or len(token) < 3:
        return False

    has_underscore = "_" in token
    has_dot = "." in token
    has_camel = bool(_CAMEL_SPLIT.search(token))
    has_acronym = bool(_ACRONYM_SPLIT.search(token))

    return has_underscore or has_dot or has_camel or has_acronym


# ---------------------------------------------------------------------------
# Text expansion
# ---------------------------------------------------------------------------


def expand_token(token: str) -> str:
    """
    Expand a single token by appending its split parts.

    The original token is preserved first (for exact matching),
    then the split parts are appended (for partial matching).

    Returns the original token unchanged if it's not a compound identifier.
    """
    if not is_compound_identifier(token):
        return token

    parts = split_identifier(token)
    if not parts or len(parts) <= 1:
        return token

    # Original + split parts
    return token + " " + " ".join(parts)


def expand_text(text: str) -> str:
    """
    Expand all compound identifiers in a text block.

    Finds tokens that look like code identifiers and appends their
    split components. Non-identifier text passes through unchanged.

    Parameters
    ----------
    text : str
        Text potentially containing code identifiers.

    Returns
    -------
    str
        Text with compound identifiers expanded.
    """
    if not text:
        return text

    seen: Set[str] = set()
    expansions: List[str] = []

    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        if token in seen:
            continue
        seen.add(token)

        if is_compound_identifier(token):
            parts = split_identifier(token)
            if parts and len(parts) > 1:
                # Add just the split parts (original is already in text)
                expansions.extend(parts)

    if not expansions:
        return text

    # Append expanded terms at the end, space-separated
    return text + " " + " ".join(expansions)


def expand_query(query: str) -> str:
    """
    Expand a search query by splitting compound identifiers.

    Each compound term becomes an OR group: the original term plus
    its components.  Non-compound terms pass through unchanged.

    Parameters
    ----------
    query : str
        Search query string.

    Returns
    -------
    str
        Expanded query with split identifier parts added.

    Examples
    --------
    >>> expand_query("getUserById error")
    'getUserById get user by id error'
    """
    if not query:
        return query

    terms = query.strip().split()
    expanded_terms: List[str] = []

    for term in terms:
        expanded_terms.append(term)
        if is_compound_identifier(term):
            parts = split_identifier(term)
            if parts and len(parts) > 1:
                expanded_terms.extend(parts)

    return " ".join(expanded_terms)
