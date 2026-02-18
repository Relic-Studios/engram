"""
engram.soul -- Code-first SOUL.md philosophy generator.

Replaces the old soul_creator.py which generated prose from emotional
seed values (OCEAN personality, VAD emotional state).  The new generator
creates a project-aware SOUL.md from technical seed values: coding
philosophy, proficiency stack, architectural patterns, review checklist,
and forbidden constructs.

The SOUL.md acts as the agent's "source of truth" — encoding the user's
coding philosophy and the project's technical constraints.  It is loaded
at boot time and injected into the context window as the IDENTITY section.

Usage:
    from engram.soul import SoulTemplate, generate_soul

    template = SoulTemplate(
        name="backend-api",
        philosophy=["Test-First", "Explicit over Implicit"],
        proficiency={"Python": "expert", "FastAPI": "expert"},
        patterns=["Repository pattern for data access", "Dependency injection"],
        review_checklist=["No bare except clauses", "All public functions typed"],
        forbidden=["eval()", "import *", "print() without logging"],
    )
    soul_text = generate_soul(template)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Seed values — the "TypeSafety, TestFirst" technical personality
# ---------------------------------------------------------------------------

DEFAULT_PHILOSOPHY = [
    "Test-First: write tests before implementation when possible",
    "Explicit over Implicit: no magic, no hidden side effects",
    "Type Safety: use type annotations everywhere",
    "Fail Fast: validate inputs early, surface errors immediately",
    "Single Responsibility: each function/class does one thing well",
]

DEFAULT_REVIEW_CHECKLIST = [
    "All public functions have type annotations",
    "All public functions have docstrings",
    "No bare except clauses — catch specific exceptions",
    "No wildcard imports",
    "Error messages are actionable and specific",
    "Cyclomatic complexity per function ≤ 10",
    "No TODO/FIXME without a tracking issue",
]

DEFAULT_FORBIDDEN = [
    "eval() / exec() — security risk, use explicit parsing",
    "from X import * — pollutes namespace, use explicit imports",
    "Bare except: — always catch specific exceptions",
    "print() for logging — use the logging module",
    "type: ignore without explanation",
    "Global mutable state",
]


# ---------------------------------------------------------------------------
# SoulTemplate
# ---------------------------------------------------------------------------


@dataclass
class SoulTemplate:
    """Configuration for generating a project SOUL.md.

    Parameters
    ----------
    name:
        Project or agent name (used in the header).
    philosophy:
        Core coding principles.  Defaults to the "TypeSafety, TestFirst"
        philosophy from the buildplan.
    proficiency:
        Language/framework proficiency map.  Keys are technology names,
        values are proficiency levels (expert, proficient, learning).
    patterns:
        Architectural patterns used in the project (e.g., "Repository
        pattern", "Event-driven architecture").
    review_checklist:
        Deterministic quality gates for code review.
    forbidden:
        Constructs that should never appear in generated code.
    custom_sections:
        Additional sections to include in the SOUL.md.  Keys are
        section titles, values are markdown content.
    """

    name: str = "engram"
    philosophy: List[str] = field(default_factory=lambda: list(DEFAULT_PHILOSOPHY))
    proficiency: Dict[str, str] = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)
    review_checklist: List[str] = field(
        default_factory=lambda: list(DEFAULT_REVIEW_CHECKLIST)
    )
    forbidden: List[str] = field(default_factory=lambda: list(DEFAULT_FORBIDDEN))
    custom_sections: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_soul(template: Optional[SoulTemplate] = None) -> str:
    """Generate a SOUL.md document from a SoulTemplate.

    Returns the full markdown text ready to write to a file.
    If no template is provided, generates a default template with
    the "TypeSafety, TestFirst" philosophy.
    """
    if template is None:
        template = SoulTemplate()

    sections: List[str] = []

    # Header
    sections.append(f"# {template.name} — Coding Philosophy\n")
    sections.append(
        "This document defines the coding philosophy, standards, and "
        "constraints for this project.  It is loaded at the start of "
        "every session to ground the agent's technical decisions.\n"
    )

    # Philosophy
    sections.append("## Philosophy\n")
    for principle in template.philosophy:
        sections.append(f"- **{principle}**")
    sections.append("")

    # Proficiency
    if template.proficiency:
        sections.append("## Proficiency\n")
        sections.append("| Technology | Level |")
        sections.append("|------------|-------|")
        for tech, level in sorted(template.proficiency.items()):
            sections.append(f"| {tech} | {level} |")
        sections.append("")

    # Architectural Patterns
    if template.patterns:
        sections.append("## Architectural Patterns\n")
        for pattern in template.patterns:
            sections.append(f"- {pattern}")
        sections.append("")

    # Code Review Checklist
    if template.review_checklist:
        sections.append("## Code Review Checklist\n")
        for check in template.review_checklist:
            sections.append(f"- [ ] {check}")
        sections.append("")

    # Forbidden Constructs
    if template.forbidden:
        sections.append("## Forbidden Constructs\n")
        sections.append(
            "The following constructs must never appear in generated code:\n"
        )
        for item in template.forbidden:
            sections.append(f"- ~~{item}~~")
        sections.append("")

    # Custom sections
    for title, content in template.custom_sections.items():
        sections.append(f"## {title}\n")
        sections.append(content)
        sections.append("")

    return "\n".join(sections)


def write_soul(
    template: Optional[SoulTemplate] = None,
    soul_dir: Optional[Path] = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Generate and write a SOUL.md file.

    Parameters
    ----------
    template:
        SoulTemplate configuration.  Uses defaults if not provided.
    soul_dir:
        Directory to write the SOUL.md into.  Creates it if needed.
    overwrite:
        If False (default), will not overwrite an existing SOUL.md.
        Set to True to force regeneration.

    Returns
    -------
    Path
        Path to the written SOUL.md file.

    Raises
    ------
    FileExistsError
        If SOUL.md already exists and overwrite is False.
    """
    if soul_dir is None:
        soul_dir = Path(".")

    soul_dir = Path(soul_dir)
    soul_dir.mkdir(parents=True, exist_ok=True)

    soul_path = soul_dir / "SOUL.md"

    if soul_path.exists() and not overwrite:
        raise FileExistsError(
            f"SOUL.md already exists at {soul_path}. Set overwrite=True to regenerate."
        )

    content = generate_soul(template)
    soul_path.write_text(content, encoding="utf-8")
    return soul_path


def load_template_from_soul(soul_path: Path) -> SoulTemplate:
    """Parse an existing SOUL.md back into a SoulTemplate.

    This is a best-effort parser — it extracts what it can from
    the markdown structure.  Useful for loading an existing SOUL.md,
    modifying it, and regenerating.
    """
    if not soul_path.exists():
        return SoulTemplate()

    text = soul_path.read_text(encoding="utf-8")
    template = SoulTemplate()

    # Extract name from header
    for line in text.split("\n"):
        if line.startswith("# "):
            # "# project-name — Coding Philosophy" -> "project-name"
            name_part = line[2:].split("—")[0].split("-")[0].strip()
            if name_part:
                template.name = name_part.split("—")[0].strip()
            break

    # Extract philosophy items
    template.philosophy = _extract_list_section(text, "## Philosophy")

    # Extract patterns
    template.patterns = _extract_list_section(text, "## Architectural Patterns")

    # Extract review checklist (strip "[ ] " prefix)
    raw_checklist = _extract_list_section(text, "## Code Review Checklist")
    template.review_checklist = [
        item.lstrip("[ ] ").lstrip("[x] ") for item in raw_checklist
    ]

    # Extract forbidden (strip "~~" wrapping)
    raw_forbidden = _extract_list_section(text, "## Forbidden Constructs")
    template.forbidden = [item.strip("~") for item in raw_forbidden]

    # Extract proficiency table
    template.proficiency = _extract_table_section(text, "## Proficiency")

    return template


def _extract_list_section(text: str, header: str) -> List[str]:
    """Extract bullet-point items from a markdown section."""
    items: List[str] = []
    in_section = False

    for line in text.split("\n"):
        if line.strip() == header:
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break  # Next section
            stripped = line.strip()
            if stripped.startswith("- "):
                # Remove "- ", "- **...**", "- [ ] ", "- ~~...~~"
                item = stripped[2:].strip()
                # Remove bold markers
                item = item.strip("*")
                items.append(item)

    return items


def _extract_table_section(text: str, header: str) -> Dict[str, str]:
    """Extract a two-column markdown table from a section."""
    result: Dict[str, str] = {}
    in_section = False
    header_passed = False

    for line in text.split("\n"):
        if line.strip() == header:
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break
            stripped = line.strip()
            if stripped.startswith("|") and "---" not in stripped:
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                if len(cells) >= 2:
                    if not header_passed:
                        header_passed = True  # Skip the header row
                        continue
                    result[cells[0]] = cells[1]

    return result
