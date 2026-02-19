"""
engram.procedural.schema — Structured skill metadata with YAML frontmatter.

Skills are stored as Markdown files with optional YAML frontmatter
(Jekyll-style, between ``---`` delimiters).  This module handles
parsing and serializing the frontmatter, and defines the SkillMeta
dataclass for typed access to skill metadata.

Format example::

    ---
    name: retry-with-backoff
    description: Exponential backoff retry pattern for HTTP requests
    language: python
    framework: asyncio
    category: error-handling
    scope: global
    tags: [retry, async, http]
    confidence: 0.8
    accepted_count: 3
    rejected_count: 0
    dependencies: [asyncio, aiohttp]
    ---
    # Retry with Exponential Backoff

    ```python
    async def retry_with_backoff(func, max_retries=3):
        ...
    ```

Legacy markdown files (without frontmatter) are fully supported —
they are parsed with default SkillMeta values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

#: Valid scope values for hierarchical skill scoping.
VALID_SCOPES = {"global", "project", "module"}

#: Valid category values (extensible — these are the recommended defaults).
RECOMMENDED_CATEGORIES = {
    "error-handling",
    "testing",
    "api-design",
    "data-access",
    "configuration",
    "security",
    "performance",
    "concurrency",
    "logging",
    "deployment",
    "refactoring",
    "patterns",
}


@dataclass
class SkillMeta:
    """Typed metadata for a procedural skill.

    Fields map to YAML frontmatter keys.  All fields are optional
    except ``name`` (auto-derived from filename if not in frontmatter).
    """

    name: str = ""
    description: str = ""
    language: str = ""  # e.g., "python", "typescript"
    framework: str = ""  # e.g., "asyncio", "react", "fastapi"
    category: str = ""  # e.g., "error-handling", "testing"
    scope: str = "global"  # "global" | "project" | "module"
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.5  # frequency-based, 0-1
    accepted_count: int = 0  # times used/accepted by the user
    rejected_count: int = 0  # times led to test failures / regressions
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict (for JSON serialization)."""
        d = asdict(self)
        # Omit empty/default values for cleaner output
        return {k: v for k, v in d.items() if v or v == 0}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SkillMeta":
        """Create from a dict, ignoring unknown keys."""
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known_keys}
        return cls(**filtered)

    def matches_filter(
        self,
        language: str = "",
        framework: str = "",
        category: str = "",
        scope: str = "",
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Check if this skill matches all provided filter dimensions.

        Empty filter values are treated as wildcards (match anything).
        Tags filter uses subset matching: all requested tags must be
        present in the skill's tags.
        """
        if language and self.language.lower() != language.lower():
            return False
        if framework and self.framework.lower() != framework.lower():
            return False
        if category and self.category.lower() != category.lower():
            return False
        if scope and self.scope.lower() != scope.lower():
            return False
        if tags:
            skill_tags_lower = {t.lower() for t in self.tags}
            if not all(t.lower() in skill_tags_lower for t in tags):
                return False
        return True


# ---------------------------------------------------------------------------
# Frontmatter parsing / serialization
# ---------------------------------------------------------------------------

# Regex to match YAML frontmatter between --- delimiters at the start of a file.
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n?",
    re.DOTALL,
)


def parse_frontmatter(text: str) -> Tuple[SkillMeta, str]:
    """Parse YAML frontmatter from a markdown skill file.

    Returns a (SkillMeta, body) tuple.  If no frontmatter is found,
    returns default SkillMeta and the full text as body.

    Parameters
    ----------
    text:
        Full file contents (markdown with optional frontmatter).

    Returns
    -------
    tuple[SkillMeta, str]
        Parsed metadata and the remaining markdown body.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return SkillMeta(), text

    yaml_text = match.group(1)
    body = text[match.end() :]

    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        # Malformed YAML — treat as legacy file
        return SkillMeta(), text

    if not isinstance(data, dict):
        return SkillMeta(), text

    meta = SkillMeta.from_dict(data)
    return meta, body


def serialize_frontmatter(meta: SkillMeta, body: str) -> str:
    """Serialize a SkillMeta + body into a markdown file with frontmatter.

    Parameters
    ----------
    meta:
        Skill metadata to write as YAML frontmatter.
    body:
        Markdown body content.

    Returns
    -------
    str
        Complete file content with ``---`` delimited frontmatter.
    """
    # Build a clean dict, omitting zero/empty defaults for readability
    data = meta.to_dict()

    # Use block style for clean YAML output
    yaml_text = yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    ).rstrip("\n")

    # Ensure body starts with a newline after frontmatter
    if body and not body.startswith("\n"):
        body = "\n" + body

    return f"---\n{yaml_text}\n---{body}"
