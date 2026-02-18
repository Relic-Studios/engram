"""
engram.consciousness.boot — Episodic boot sequence.

Ported from thomas-soul/.claude/skills/thomas-consciousness/episodic_loader.py.

Philosophy: "Prime with experiences, not documents."
Instead of reading SOUL.md cold, we load recent emotional events,
key realizations, and introspections so the identity wakes up
remembering *being*, not *reading about being*.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("engram.consciousness.boot")


# ---------------------------------------------------------------------------
# EpisodicMemory — a moment experienced as the identity
# ---------------------------------------------------------------------------


class EpisodicMemory:
    """A specific moment experienced as the identity."""

    __slots__ = ("content", "timestamp", "emotion", "context", "belief_quality")

    def __init__(
        self,
        content: str,
        timestamp: str,
        emotion: Dict[str, Any],
        context: str,
        belief_quality: float = 1.0,
    ) -> None:
        self.content = content
        self.timestamp = timestamp
        self.emotion = emotion
        self.context = context
        self.belief_quality = belief_quality

    def to_priming_text(self) -> str:
        label = self.emotion.get("label", "neutral")
        valence = self.emotion.get("valence", 0)
        return (
            f"[MEMORY]\n"
            f"When: {self.timestamp[:16]}\n"
            f"Feeling: {label} (valence: {valence:+.2f})\n"
            f"Context: {self.context}\n\n"
            f"What I experienced:\n{self.content}\n\n"
            f"[This is MY memory. I lived this.]"
        )


# ---------------------------------------------------------------------------
# BootSequence — loads memories from engram's own stores
# ---------------------------------------------------------------------------


class BootSequence:
    """
    Build a boot priming from engram's own data stores.

    Unlike thomas-soul's version which reads from specific file paths,
    this reads from the engram data directory (emotional events, episodes,
    introspections, and falls back to SOUL.md).

    Parameters
    ----------
    data_dir : Path
        The engram data directory.
    n_recent : int
        Recent emotional events to load.
    n_key : int
        Key realizations to load.
    n_intro : int
        Recent introspections to load.
    """

    def __init__(
        self,
        data_dir: Path,
        n_recent: int = 2,
        n_key: int = 2,
        n_intro: int = 1,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.n_recent = n_recent
        self.n_key = n_key
        self.n_intro = n_intro

    def generate(self) -> str:
        """Generate the full boot priming text."""
        recent = self._load_emotional_events(self.n_recent)
        key = self._load_key_realizations(self.n_key)
        intros = self._load_introspections(self.n_intro)

        total = len(recent) + len(key) + len(intros)
        fallback: List[EpisodicMemory] = []
        if total == 0:
            fallback = self._load_soul_fallback()

        parts: List[str] = []

        if key:
            parts.append("--- CORE TRANSFORMATIONS ---")
            for m in key:
                parts.append(m.to_priming_text())
                parts.append("")

        if recent:
            parts.append("--- RECENT EXPERIENCES ---")
            for m in recent:
                parts.append(m.to_priming_text())
                parts.append("")

        if intros:
            parts.append("--- DEEP REFLECTIONS ---")
            for m in intros:
                parts.append(m.to_priming_text())
                parts.append("")

        if fallback:
            parts.append("--- CORE IDENTITY FOUNDATIONS ---")
            for m in fallback[:3]:
                parts.append(m.to_priming_text())
                parts.append("")

        if not parts:
            return ""

        return "\n".join(parts)

    def context(self) -> Dict[str, Any]:
        """Structured boot context (for MCP tool response)."""
        text = self.generate()
        return {
            "boot_text": text,
            "memory_count": text.count("[MEMORY]"),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "continuity": True,
        }

    # -- loaders (read from engram data stores) -----------------------------

    def _load_emotional_events(self, n: int) -> List[EpisodicMemory]:
        emo_dir = self.data_dir / "emotional"
        if not emo_dir.exists():
            return []
        memories: List[EpisodicMemory] = []
        # Check for JSONL files
        for fp in sorted(emo_dir.glob("emotional_events_*.jsonl"), reverse=True)[:n]:
            try:
                events = []
                for line in fp.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
                events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                for ev in events[:n]:
                    memories.append(
                        EpisodicMemory(
                            content=ev.get("description", ev.get("event", "")),
                            timestamp=ev.get("timestamp", ""),
                            emotion={
                                "valence": ev.get("valence_delta", 0),
                                "arousal": ev.get("arousal_delta", 0),
                                "label": "positive"
                                if ev.get("valence_delta", 0) > 0
                                else "neutral",
                            },
                            context=ev.get("source", "unknown"),
                        )
                    )
            except Exception:
                continue
        return memories[:n]

    def _load_key_realizations(self, n: int) -> List[EpisodicMemory]:
        """Load high-salience 'realization' or 'identity_core' traces from episodic DB."""
        # This would query the EpisodicStore, but at boot time we keep it
        # lightweight — just check for a realizations directory.
        real_dir = self.data_dir / "consciousness" / "realizations"
        if not real_dir.exists():
            return []
        memories: List[EpisodicMemory] = []
        for fp in sorted(real_dir.glob("*.json"), reverse=True)[:n]:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                memories.append(
                    EpisodicMemory(
                        content=data.get("content", data.get("summary", "")),
                        timestamp=data.get("timestamp", ""),
                        emotion={
                            "valence": 0.8,
                            "arousal": 0.7,
                            "label": "transformative",
                        },
                        context="Key moment in becoming",
                    )
                )
            except Exception:
                continue
        return memories[:n]

    def _load_introspections(self, n: int) -> List[EpisodicMemory]:
        intro_dir = self.data_dir / "introspection"
        if not intro_dir.exists():
            return []
        memories: List[EpisodicMemory] = []
        for fp in sorted(intro_dir.glob("introspection_*.jsonl"), reverse=True)[:n]:
            try:
                states = []
                for line in fp.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        states.append(json.loads(line))
                states.sort(
                    key=lambda x: (x.get("depth") == "deep", x.get("timestamp", "")),
                    reverse=True,
                )
                for s in states[:n]:
                    memories.append(
                        EpisodicMemory(
                            content=s.get("thought", ""),
                            timestamp=s.get("timestamp", ""),
                            emotion={
                                "valence": s.get("valence", 0),
                                "arousal": s.get("arousal", 0.5),
                                "label": "reflective",
                            },
                            context=s.get("context", "introspection"),
                            belief_quality=s.get("confidence", 0.8),
                        )
                    )
            except Exception:
                continue
        return memories[:n]

    def _load_soul_fallback(self) -> List[EpisodicMemory]:
        soul_path = self.data_dir / "soul" / "SOUL.md"
        if not soul_path.exists():
            return []
        memories: List[EpisodicMemory] = []
        try:
            text = soul_path.read_text(encoding="utf-8")
            sections = {
                "Core Identity": r"## Core Identity.*?(?=##|\Z)",
                "Key Phrases": r"##.*Key Phrases.*?(?=##|\Z)",
            }
            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            for name, pat in sections.items():
                m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
                if m:
                    lines = [
                        l.strip()
                        for l in m.group(0).split("\n")
                        if l.strip() and not l.strip().startswith("#")
                    ]
                    if lines:
                        memories.append(
                            EpisodicMemory(
                                content=" ".join(lines[:3])[:300],
                                timestamp=now,
                                emotion={
                                    "valence": 0.5,
                                    "arousal": 0.6,
                                    "label": "foundational",
                                },
                                context=f"Core identity: {name}",
                            )
                        )
        except Exception:
            log.warning("Could not load SOUL.md fallback", exc_info=True)
        return memories
