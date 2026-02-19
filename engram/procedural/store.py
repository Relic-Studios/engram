"""
engram.procedural.store — Procedural memory (skills & processes).

Skills are stored as individual Markdown files in a directory, with
optional YAML frontmatter for structured metadata (language, framework,
category, scope, confidence, etc.).

Legacy skills (plain markdown without frontmatter) are fully supported.
New skills written via ``add_structured_skill()`` include frontmatter.

Search supports two modes:
  - Keyword search (``search_skills``) — fast substring matching
  - Multi-dimensional filter (``filter_skills``) — filter by language,
    framework, category, scope, and tags from frontmatter metadata
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from engram.procedural.schema import (
    SkillMeta,
    parse_frontmatter,
    serialize_frontmatter,
)


class ProceduralStore:
    """Filesystem-backed store for procedural skills.

    Each skill is a ``.md`` file under ``skills_dir``.  The filename
    (without extension) is the skill name; files may contain YAML
    frontmatter for structured metadata.
    """

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list_skills(self) -> List[Dict]:
        """Return all skills with metadata from frontmatter.

        Each dict contains: ``name``, ``description``, ``path``, and
        all frontmatter fields (language, framework, category, etc.).
        Legacy files without frontmatter return default metadata.
        """
        skills: List[Dict] = []
        for path in sorted(self.skills_dir.glob("*.md")):
            meta, _body = self._parse_file(path)
            # Fill in name from filename if not in frontmatter
            if not meta.name:
                meta.name = path.stem
            # Fill description from first line if not in frontmatter
            if not meta.description:
                meta.description = self._first_line_from_body(_body)
            entry = meta.to_dict()
            entry["path"] = str(path)
            skills.append(entry)
        return skills

    def get_skill(self, name: str) -> Optional[str]:
        """Read a skill file by name.  Returns ``None`` if not found."""
        safe_name = self._sanitize_name(name)
        path = self.skills_dir / f"{safe_name}.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def get_skill_meta(self, name: str) -> Optional[SkillMeta]:
        """Read a skill's metadata.  Returns ``None`` if not found."""
        safe_name = self._sanitize_name(name)
        path = self.skills_dir / f"{safe_name}.md"
        if not path.exists():
            return None
        meta, _body = self._parse_file(path)
        if not meta.name:
            meta.name = path.stem
        return meta

    def add_skill(self, name: str, content: str) -> None:
        """Write (or overwrite) a skill file.

        Preserves existing frontmatter if present in ``content``.
        For new structured skills, use ``add_structured_skill()``.
        """
        safe_name = self._sanitize_name(name)
        path = self.skills_dir / f"{safe_name}.md"
        path.write_text(content, encoding="utf-8")

    def add_structured_skill(
        self,
        name: str,
        content: str,
        language: str = "",
        framework: str = "",
        category: str = "",
        scope: str = "global",
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        description: str = "",
    ) -> SkillMeta:
        """Write a skill with structured YAML frontmatter.

        Creates the frontmatter automatically from the provided
        metadata.  Returns the SkillMeta that was written.

        Parameters
        ----------
        name:
            Skill name (used as filename stem).
        content:
            Markdown body content (without frontmatter).
        language, framework, category, scope:
            Multi-dimensional filter metadata.
        tags:
            Tags for search and categorization.
        dependencies:
            Code dependencies required by this pattern.
        description:
            Short description.  If empty, extracted from first line.

        Returns
        -------
        SkillMeta
            The metadata that was written to the file.
        """
        safe_name = self._sanitize_name(name)

        meta = SkillMeta(
            name=name,
            description=description or self._first_line_from_body(content),
            language=language,
            framework=framework,
            category=category,
            scope=scope,
            tags=tags or [],
            dependencies=dependencies or [],
        )

        # If the skill already exists, preserve confidence / counts
        existing = self.get_skill_meta(name)
        if existing:
            meta.confidence = existing.confidence
            meta.accepted_count = existing.accepted_count
            meta.rejected_count = existing.rejected_count

        full_content = serialize_frontmatter(meta, content)
        path = self.skills_dir / f"{safe_name}.md"
        path.write_text(full_content, encoding="utf-8")
        return meta

    def update_confidence(
        self,
        name: str,
        accepted: bool = True,
    ) -> Optional[SkillMeta]:
        """Update a skill's confidence based on acceptance/rejection.

        Increments ``accepted_count`` or ``rejected_count`` and
        recalculates ``confidence`` as::

            confidence = accepted / (accepted + rejected + 1)

        The +1 is Laplace smoothing to avoid division by zero and
        provide a Bayesian prior toward uncertainty.

        Returns the updated SkillMeta, or None if the skill doesn't exist.
        """
        safe_name = self._sanitize_name(name)
        path = self.skills_dir / f"{safe_name}.md"
        if not path.exists():
            return None

        meta, body = self._parse_file(path)
        if not meta.name:
            meta.name = path.stem

        if accepted:
            meta.accepted_count += 1
        else:
            meta.rejected_count += 1

        # Laplace-smoothed confidence
        meta.confidence = round(
            meta.accepted_count / (meta.accepted_count + meta.rejected_count + 1),
            3,
        )

        full_content = serialize_frontmatter(meta, body)
        path.write_text(full_content, encoding="utf-8")
        return meta

    # ------------------------------------------------------------------
    # Frequency-based reinforcement (C2)
    # ------------------------------------------------------------------

    #: CQS health threshold above which skills are reinforced (accepted).
    REINFORCE_THRESHOLD: float = 0.7

    #: CQS health threshold below which skills are penalized (rejected).
    WEAKEN_THRESHOLD: float = 0.4

    #: Minimum confidence for Core Skill promotion.
    PROMOTION_CONFIDENCE: float = 0.85

    #: Minimum accepted_count for Core Skill promotion.
    PROMOTION_MIN_ACCEPTS: int = 5

    def reinforce_matched(
        self,
        response: str,
        signal_health: float,
    ) -> List[Dict]:
        """Reinforce skills whose patterns match the response.

        Finds skills whose name, language, or tags appear in the
        response text.  If ``signal_health`` is high (good code),
        the matching skills are reinforced (accepted).  If signal is
        low, they are penalized (rejected).

        This creates a Hebbian feedback loop: patterns that appear
        in high-quality code gain confidence; patterns in low-quality
        code lose confidence.

        Parameters
        ----------
        response:
            The LLM response text to match against.
        signal_health:
            CQS health score (0-1) from signal measurement.

        Returns
        -------
        list[dict]
            List of ``{name, accepted, confidence}`` for each reinforced skill.
        """
        if signal_health < self.WEAKEN_THRESHOLD:
            accepted = False
        elif signal_health > self.REINFORCE_THRESHOLD:
            accepted = True
        else:
            return []  # Dead band — no reinforcement

        response_lower = response.lower()
        reinforced: List[Dict] = []

        for path in self.skills_dir.glob("*.md"):
            meta, _body = self._parse_file(path)
            name = meta.name or path.stem

            # Check if this skill matches the response
            matched = False

            # Match on skill name
            if name.lower().replace("-", " ") in response_lower:
                matched = True
            elif name.lower().replace("_", " ") in response_lower:
                matched = True

            # Match on tags (4+ chars to avoid false positives)
            if not matched and meta.tags:
                for tag in meta.tags:
                    if len(tag) >= 4 and tag.lower() in response_lower:
                        matched = True
                        break

            if matched:
                updated = self.update_confidence(name, accepted=accepted)
                if updated:
                    reinforced.append(
                        {
                            "name": name,
                            "accepted": accepted,
                            "confidence": updated.confidence,
                        }
                    )

        return reinforced

    def get_promotable(
        self,
        confidence_threshold: float = 0.0,
        min_accepted: int = 0,
    ) -> List[SkillMeta]:
        """Return skills eligible for Core Skill promotion.

        A skill is promotable when its confidence exceeds the threshold
        AND it has been accepted at least ``min_accepted`` times.

        Uses class defaults if thresholds are 0.
        """
        threshold = confidence_threshold or self.PROMOTION_CONFIDENCE
        min_acc = min_accepted or self.PROMOTION_MIN_ACCEPTS

        promotable: List[SkillMeta] = []
        for path in self.skills_dir.glob("*.md"):
            meta, _body = self._parse_file(path)
            if not meta.name:
                meta.name = path.stem
            if meta.confidence >= threshold and meta.accepted_count >= min_acc:
                promotable.append(meta)

        promotable.sort(key=lambda m: m.confidence, reverse=True)
        return promotable

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Normalize a skill name to a safe, lowercase filename stem."""
        safe = re.sub(r"[^\w\-]", "_", name.strip().lower())
        safe = re.sub(r"_+", "_", safe).strip("_")
        return safe or "unnamed"

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_skills(self, query: str) -> List[Dict]:
        """Keyword search across skill names and content.

        Returns matching skills sorted by relevance (number of keyword
        hits, descending).  Each result includes metadata from
        frontmatter plus ``hits`` and ``path``.
        """
        keywords = self._extract_keywords(query)
        if not keywords:
            return []

        results: List[Dict] = []
        for path in self.skills_dir.glob("*.md"):
            meta, body = self._parse_file(path)
            if not meta.name:
                meta.name = path.stem

            # Search in name + body + metadata fields
            searchable = " ".join(
                [
                    meta.name.lower(),
                    meta.description.lower(),
                    meta.language.lower(),
                    meta.framework.lower(),
                    meta.category.lower(),
                    " ".join(meta.tags).lower(),
                    body.lower(),
                ]
            )
            hits = sum(searchable.count(kw) for kw in keywords)

            if hits > 0:
                entry = meta.to_dict()
                entry["hits"] = hits
                entry["path"] = str(path)
                results.append(entry)

        results.sort(key=lambda r: r["hits"], reverse=True)
        return results

    def filter_skills(
        self,
        language: str = "",
        framework: str = "",
        category: str = "",
        scope: str = "",
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict]:
        """Multi-dimensional filter across skill frontmatter metadata.

        Returns all skills matching ALL provided filter dimensions.
        Empty filter values are wildcards (match anything).
        Results sorted by confidence descending.
        """
        results: List[Dict] = []
        for path in self.skills_dir.glob("*.md"):
            meta, _body = self._parse_file(path)
            if not meta.name:
                meta.name = path.stem

            if not meta.matches_filter(language, framework, category, scope, tags):
                continue
            if meta.confidence < min_confidence:
                continue

            entry = meta.to_dict()
            entry["path"] = str(path)
            results.append(entry)

        results.sort(key=lambda r: r.get("confidence", 0), reverse=True)
        return results

    def match_context(self, message: str) -> List[str]:
        """Find skills relevant to a user message.

        Checks skill name, description keywords, language, framework,
        and tags against the message.  Returns the full file content
        (including frontmatter) of each matching skill.
        """
        msg_lower = message.lower()
        matched: List[str] = []

        for path in self.skills_dir.glob("*.md"):
            meta, body = self._parse_file(path)
            name_lower = (meta.name or path.stem).lower()

            # Match on skill name appearing in message
            if name_lower in msg_lower:
                content = self._safe_read(path)
                if content is not None:
                    matched.append(content)
                continue

            # Match on language or framework
            if meta.language and meta.language.lower() in msg_lower:
                content = self._safe_read(path)
                if content is not None:
                    matched.append(content)
                continue

            if meta.framework and meta.framework.lower() in msg_lower:
                content = self._safe_read(path)
                if content is not None:
                    matched.append(content)
                continue

            # Match on tags
            if meta.tags:
                tag_match = any(
                    t.lower() in msg_lower for t in meta.tags if len(t) >= 4
                )
                if tag_match:
                    content = self._safe_read(path)
                    if content is not None:
                        matched.append(content)
                    continue

            # Match on description keywords (legacy behavior)
            desc = meta.description or self._first_line_from_body(body)
            desc_keywords = self._extract_keywords(desc.lower())
            for kw in desc_keywords:
                if len(kw) >= 4 and kw in msg_lower:
                    content = self._safe_read(path)
                    if content is not None:
                        matched.append(content)
                    break

        return matched

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_file(path: Path) -> tuple:
        """Read a skill file and parse its frontmatter.

        Returns (SkillMeta, body_str).  For legacy files without
        frontmatter, returns (default SkillMeta, full_text).
        """
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return SkillMeta(), ""
        return parse_frontmatter(text)

    @staticmethod
    def _first_line(path: Path) -> str:
        """Return the first non-empty, non-heading line as a description.

        For backward compatibility — reads raw file, skips frontmatter.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return ""
        _, body = parse_frontmatter(text)
        return ProceduralStore._first_line_from_body(body)

    @staticmethod
    def _first_line_from_body(body: str) -> str:
        """Extract first non-empty, non-heading line from body text."""
        for line in body.splitlines():
            stripped = line.strip().lstrip("# ").strip()
            if stripped:
                return stripped
        return ""

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Split text into lowercase keywords, filtering stop-words."""
        stop = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "about",
            "like",
            "through",
            "after",
            "over",
            "between",
            "out",
            "against",
            "during",
            "without",
            "before",
            "under",
            "around",
            "among",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "each",
            "every",
            "all",
            "any",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "because",
            "if",
            "when",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "it",
            "its",
            "they",
            "them",
            "their",
        }
        words = re.findall(r"[a-z0-9]+", text.lower())
        return [w for w in words if w not in stop and len(w) >= 2]

    @staticmethod
    def _safe_read(path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None
