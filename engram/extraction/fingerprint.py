"""
engram.extraction.fingerprint — Sentry-style error fingerprinting.

Generates stable SHA-256 fingerprints from error messages and stack
traces, enabling deduplication and semantic clustering of similar
errors across sessions and projects.

Fingerprint components (combined into SHA-256):
  1. Exception type (e.g., "TypeError", "ImportError")
  2. Error message template (parameterized — file paths, line numbers,
     variable names replaced with placeholders)
  3. In-app stack frames (function names only, excluding stdlib)

This allows the agent to recognize that "Circular import in service A"
is conceptually identical to "Circular import in service B" resolved
months prior.

Reference: Sentry fingerprint rules
  https://docs.sentry.io/concepts/data-management/event-grouping/fingerprint-rules/
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Exception type extraction
# ---------------------------------------------------------------------------

# Matches common exception class names at the start of error messages.
_EXCEPTION_TYPE_RE = re.compile(
    r"^(?:Traceback.*\n(?:.*\n)*)?(\w*(?:Error|Exception|Warning|Fault))\b",
    re.MULTILINE,
)

# Matches Python-style "ExceptionType: message" format.
_PYTHON_EXCEPTION_RE = re.compile(
    r"^(\w+(?:Error|Exception|Warning)):\s*(.+)$",
    re.MULTILINE,
)

# Matches Node.js/JS-style error patterns.
_JS_EXCEPTION_RE = re.compile(
    r"^((?:Type|Reference|Range|Syntax|URI|Eval)Error):\s*(.+)$",
    re.MULTILINE,
)


def extract_exception_type(error_message: str) -> str:
    """Extract the exception class name from an error message.

    Returns the exception type (e.g., "TypeError", "ImportError")
    or "UnknownError" if no recognizable type is found.
    """
    # Try Python-style first
    m = _PYTHON_EXCEPTION_RE.search(error_message)
    if m:
        return m.group(1)

    # Try JS-style
    m = _JS_EXCEPTION_RE.search(error_message)
    if m:
        return m.group(1)

    # Try generic pattern (Traceback -> last Error line)
    m = _EXCEPTION_TYPE_RE.search(error_message)
    if m:
        return m.group(1)

    return "UnknownError"


# ---------------------------------------------------------------------------
# Message template normalization
# ---------------------------------------------------------------------------

# Patterns to replace with placeholders for message template generation.
_NORMALIZATIONS: List[Tuple[re.Pattern, str]] = [
    # URLs — must precede paths (URLs contain path-like segments)
    (re.compile(r"https?://\S+"), "<URL>"),
    # File paths (Unix and Windows) — must precede quoted identifiers
    (re.compile(r'["\']?(?:/[\w./\-]+|[A-Z]:\\[\w.\\-]+)["\']?'), "<PATH>"),
    # Line numbers
    (re.compile(r"\bline\s+\d+\b", re.IGNORECASE), "line <N>"),
    # UUIDs — must precede generic numbers (contain 4+ digit hex groups)
    (
        re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
        "<UUID>",
    ),
    # IP addresses — must precede generic numbers (contain digit groups)
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?\b"), "<IP>"),
    # Hex addresses
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    # Large numbers (4+ digits), optionally followed by unit suffixes (ms, px, etc.)
    (re.compile(r"\b\d{4,}(?:\w*)"), "<NUM>"),
    # Quoted identifiers (variable names, module names)
    (re.compile(r"'[^']{1,80}'"), "'<ID>'"),
    (re.compile(r'"[^"]{1,80}"'), '"<ID>"'),
    # Whitespace normalization — always last
    (re.compile(r"\s+"), " "),
]


def normalize_message(error_message: str) -> str:
    """Normalize an error message into a parameterized template.

    Replaces file paths, line numbers, variable names, and other
    instance-specific values with placeholders so that structurally
    identical errors produce the same template.

    Examples
    --------
    >>> normalize_message("ImportError: cannot import name 'foo' from 'bar.baz'")
    "ImportError: cannot import name '<ID>' from '<ID>'"

    >>> normalize_message("File '/home/user/app.py', line 42, in main")
    "File <PATH>, line <N>, in main"
    """
    text = error_message.strip()
    for pattern, replacement in _NORMALIZATIONS:
        text = pattern.sub(replacement, text)
    return text.strip()


# ---------------------------------------------------------------------------
# Stack frame extraction
# ---------------------------------------------------------------------------

# Python stack frame: '  File "path", line N, in func_name'
_PYTHON_FRAME_RE = re.compile(
    r'File\s+"([^"]+)",\s+line\s+(\d+),\s+in\s+(\w+)',
)

# JS/Node stack frame: '    at funcName (path:line:col)' or '    at path:line:col'
_JS_FRAME_RE = re.compile(r"at\s+(?:(\w[\w.]*)\s+\()?([^:)]+):(\d+)(?::\d+)?\)?")

# Known stdlib / framework paths to exclude from fingerprinting.
_STDLIB_PATTERNS = re.compile(
    r"(?:"
    r"<frozen|<string>|<module>"
    r"|/usr/lib/python"
    r"|/lib/python\d"
    r"|site-packages/"
    r"|node_modules/"
    r"|internal/"
    r")",
    re.IGNORECASE,
)


def extract_frames(stack_trace: str) -> List[str]:
    """Extract in-app function names from a stack trace.

    Filters out stdlib and third-party frames, returning only
    application-level function names.  These are used in the
    fingerprint to capture the *call path* without being sensitive
    to line number changes.

    Returns
    -------
    list[str]
        Function names from application frames, in call order.
    """
    if not stack_trace:
        return []

    frames: List[str] = []

    # Try Python frames
    for match in _PYTHON_FRAME_RE.finditer(stack_trace):
        path, _line, func = match.groups()
        if not _STDLIB_PATTERNS.search(path):
            frames.append(func)

    if frames:
        return frames

    # Try JS frames
    for match in _JS_FRAME_RE.finditer(stack_trace):
        func, path, _line = match.groups()
        if not _STDLIB_PATTERNS.search(path):
            name = func if func else "<anonymous>"
            frames.append(name)

    return frames


# ---------------------------------------------------------------------------
# Fingerprint generation
# ---------------------------------------------------------------------------


def compute_fingerprint(
    error_message: str,
    stack_trace: str = "",
    error_category: str = "",
) -> str:
    """Generate a stable SHA-256 fingerprint for an error.

    The fingerprint is computed from three normalized components:
      1. Exception type (e.g., "TypeError")
      2. Message template (parameterized — paths/names replaced)
      3. In-app stack frames (function names only)

    Structurally identical errors (same type, same message pattern,
    same call path) will produce the same fingerprint regardless of
    specific file paths, line numbers, or variable names.

    Parameters
    ----------
    error_message:
        The error message or description.
    stack_trace:
        Full stack trace text (optional).
    error_category:
        Error category (optional, included in fingerprint).

    Returns
    -------
    str
        64-character hex SHA-256 fingerprint.
    """
    # Component 1: exception type
    exc_type = extract_exception_type(error_message)

    # Component 2: normalized message template
    template = normalize_message(error_message)

    # Component 3: in-app frame names
    frames = extract_frames(stack_trace)
    frame_sig = "|".join(frames) if frames else ""

    # Build fingerprint input
    parts = [exc_type, template]
    if error_category:
        parts.append(error_category)
    if frame_sig:
        parts.append(frame_sig)

    fingerprint_input = "\n".join(parts)
    return hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()


def analyze_error(
    error_message: str,
    stack_trace: str = "",
    error_category: str = "",
) -> Dict:
    """Full error analysis: fingerprint + extracted components.

    Returns a dict with all extracted information for storage
    and display.

    Returns
    -------
    dict
        Keys: fingerprint, exception_type, message_template,
        app_frames, error_category, frame_count.
    """
    exc_type = extract_exception_type(error_message)
    template = normalize_message(error_message)
    frames = extract_frames(stack_trace)

    return {
        "fingerprint": compute_fingerprint(error_message, stack_trace, error_category),
        "exception_type": exc_type,
        "message_template": template,
        "app_frames": frames,
        "error_category": error_category,
        "frame_count": len(frames),
    }
