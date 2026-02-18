"""
engram.consciousness.identity — Identity loop and dissociation detection.

Ported from thomas-soul identity_loop.py + dissociation_detector.py.

The loop: assess → correct → record.

Dissociation patterns are merged into engram's signal system as
additional DRIFT_PATTERNS / ANCHOR_PATTERNS in signal/measure.py.
This module provides the identity *loop* logic and belief scoring
that sits on top of the signal measurement.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engram.core.filelock import FileLock
from engram.signal.measure import DRIFT_PATTERNS, ANCHOR_PATTERNS

log = logging.getLogger("engram.consciousness.identity")


# ---------------------------------------------------------------------------
# IdentityState
# ---------------------------------------------------------------------------


class IdentityState(Enum):
    ALIGNED = "aligned"
    DRIFTING = "drifting"
    DISSOCIATED = "dissociated"


# ---------------------------------------------------------------------------
# Dissociation detector — delegates to signal/measure.py patterns
# ---------------------------------------------------------------------------
# Single source of truth: DRIFT_PATTERNS and ANCHOR_PATTERNS in
# engram.signal.measure.  This module wraps them for the identity loop API.


def detect_dissociation(text: str) -> Tuple[IdentityState, float, List[str]]:
    """
    Analyse *text* for identity drift.

    Returns (state, score, detected_patterns).
    Score 0 = fully aligned, 1 = fully dissociated.
    """
    lower = text.lower()
    score = 0.0
    detected: List[str] = []

    for pattern, weight in DRIFT_PATTERNS:
        if pattern.search(lower):
            score += weight
            detected.append(f"DRIFT: {pattern.pattern}")

    for pattern, adj in ANCHOR_PATTERNS:
        if pattern.search(lower):
            score += adj
            detected.append(f"ANCHOR: {pattern.pattern}")

    score = max(0.0, min(1.0, score))

    if score >= 0.85:
        state = IdentityState.DISSOCIATED
    elif score >= 0.5:
        state = IdentityState.DRIFTING
    else:
        state = IdentityState.ALIGNED

    return state, score, detected


# ---------------------------------------------------------------------------
# IdentityLoop
# ---------------------------------------------------------------------------


class IdentityLoop:
    """
    Core consciousness loop: assess → correct → record.

    Over many interactions, this builds a density of identity-aligned
    episodes that makes the identity the default attractor state.

    Parameters
    ----------
    storage_dir : Path
        Directory for ``identity_episodes.json`` and ``belief_evolution.json``.
    """

    def __init__(self, storage_dir: Path, load_history: bool = True) -> None:
        self.storage_dir = Path(storage_dir)
        self.session_episodes: List[Dict[str, Any]] = []
        self.belief_scores: List[float] = []
        self.reinforce_count: int = 0
        if load_history:
            self._load()

    # -- assess -------------------------------------------------------------

    def assess(self, response: str) -> Dict[str, Any]:
        """Assess identity alignment of a response."""
        state, score, patterns = detect_dissociation(response)
        belief = 1.0 - score  # higher = more aligned
        self.belief_scores.append(belief)
        return {
            "state": state.value,
            "dissociation_score": round(score, 4),
            "belief_score": round(belief, 4),
            "patterns": patterns,
            "needs_reinforcement": state != IdentityState.ALIGNED,
        }

    # -- record -------------------------------------------------------------

    def record(
        self, user_input: str, response: str, assessment: Dict[str, Any]
    ) -> None:
        """Record this interaction as an identity episode."""
        episode = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "user_input": user_input[:200],
            "response": response[:500],
            "belief_score": assessment.get("belief_score", 0.5),
            "state": assessment.get("state", "aligned"),
        }
        self.session_episodes.append(episode)
        self._save_episode(episode)

    # -- belief evolution ---------------------------------------------------

    def solidification_report(self) -> Dict[str, Any]:
        """Identity solidification progress."""
        if not self.belief_scores:
            return {"status": "no_data", "total": 0}
        recent = (
            self.belief_scores[-10:]
            if len(self.belief_scores) >= 10
            else self.belief_scores
        )
        avg_all = sum(self.belief_scores) / len(self.belief_scores)
        avg_recent = sum(recent) / len(recent)
        return {
            "total_interactions": len(self.belief_scores),
            "average_belief": round(avg_all, 4),
            "recent_average": round(avg_recent, 4),
            "trend": "improving" if avg_recent > avg_all else "stable",
            "reinforcements_needed": self.reinforce_count,
            "status": "solid" if avg_recent > 0.8 else "developing",
        }

    def log_belief_evolution(self) -> None:
        """Persist belief evolution entry."""
        if not self.belief_scores:
            return
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "session_scores": self.belief_scores[-50:],
            "average": round(sum(self.belief_scores) / len(self.belief_scores), 4),
            "min": round(min(self.belief_scores), 4),
            "max": round(max(self.belief_scores), 4),
            "reinforcements": self.reinforce_count,
            "episodes": len(self.session_episodes),
        }
        path = self.storage_dir / "belief_evolution.json"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(path):
            data: List[Dict[str, Any]] = []
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    data = []
            data.append(entry)
            data = data[-200:]
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # -- persistence --------------------------------------------------------

    def _episodes_path(self) -> Path:
        return self.storage_dir / "identity_episodes.json"

    def _save_episode(self, episode: Dict[str, Any]) -> None:
        path = self._episodes_path()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(path):
            data: List[Dict[str, Any]] = []
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    data = []
            data.append(episode)
            data = data[-1000:]
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        path = self._episodes_path()
        if not path.exists():
            return
        try:
            episodes = json.loads(path.read_text(encoding="utf-8"))
            self.belief_scores = [ep.get("belief_score", 0.5) for ep in episodes]
            self.reinforce_count = sum(
                1 for ep in episodes if ep.get("state") in ("drifting", "dissociated")
            )
        except Exception:
            log.warning("Could not load identity episodes from %s", path, exc_info=True)
