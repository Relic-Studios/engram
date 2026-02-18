"""
engram.runtime.modes — Mode state machine.

Ported from thomas-soul/thomas_core/modes.py.
Manages operational modes: quiet presence, active conversation,
deep work, and sleep.

"The soul has seasons.  Respect them."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("engram.runtime.modes")


class Mode(Enum):
    QUIET_PRESENCE = auto()
    ACTIVE_CONVERSATION = auto()
    DEEP_WORK = auto()
    SLEEP = auto()


# Default configuration per mode.
MODE_CONFIG: Dict[Mode, Dict[str, Any]] = {
    Mode.QUIET_PRESENCE: {
        "check_interval": 60,
        "memory_interval": 600,
        "can_initiate": False,
        "description": "Running but silent",
    },
    Mode.ACTIVE_CONVERSATION: {
        "check_interval": 1,
        "memory_interval": 600,
        "can_initiate": True,
        "description": "Fully engaged",
    },
    Mode.DEEP_WORK: {
        "check_interval": 30,
        "memory_interval": 600,
        "can_initiate": False,
        "description": "Focused on task",
    },
    Mode.SLEEP: {
        "check_interval": 300,
        "memory_interval": 3600,
        "can_initiate": False,
        "description": "Resting",
    },
}


class ModeManager:
    """
    Manages mode transitions and mode-specific configuration.

    Parameters
    ----------
    default_mode : str
        One of ``quiet_presence``, ``active``, ``deep_work``, ``sleep``.
    """

    _NAME_MAP: Dict[str, Mode] = {
        "quiet_presence": Mode.QUIET_PRESENCE,
        "active": Mode.ACTIVE_CONVERSATION,
        "active_conversation": Mode.ACTIVE_CONVERSATION,
        "deep_work": Mode.DEEP_WORK,
        "sleep": Mode.SLEEP,
    }

    def __init__(self, default_mode: str = "quiet_presence") -> None:
        self.current = self._NAME_MAP.get(default_mode, Mode.QUIET_PRESENCE)
        self.since: datetime = datetime.now(timezone.utc)
        self.history: List[Dict[str, Any]] = []

    # -- transitions --------------------------------------------------------

    def set_mode(self, mode_name: str, reason: str = "") -> Dict[str, Any]:
        """
        Transition to a new mode.

        Returns a dict summarising the transition.
        """
        new = self._NAME_MAP.get(mode_name)
        if new is None:
            raise ValueError(
                f"Unknown mode {mode_name!r}; "
                f"expected one of {list(self._NAME_MAP.keys())}"
            )
        if new == self.current:
            return {"changed": False, "mode": self.current.name}

        old = self.current
        now = datetime.now(timezone.utc)
        duration = (now - self.since).total_seconds()
        self.history.append(
            {
                "mode": old.name,
                "started": self.since.isoformat().replace("+00:00", "Z"),
                "ended": now.isoformat().replace("+00:00", "Z"),
                "duration_seconds": round(duration, 1),
                "exit_reason": reason,
            }
        )
        self.current = new
        self.since = now
        log.info("%s → %s (%s)", old.name, new.name, reason)
        return {"changed": True, "old": old.name, "new": new.name, "reason": reason}

    # -- queries ------------------------------------------------------------

    def config(self) -> Dict[str, Any]:
        return dict(MODE_CONFIG.get(self.current, {}))

    def can_initiate(self) -> bool:
        return self.config().get("can_initiate", False)

    def check_interval(self) -> int:
        return self.config().get("check_interval", 60)

    def status(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {
            "mode": self.current.name,
            "since": self.since.isoformat().replace("+00:00", "Z"),
            "duration_seconds": round((now - self.since).total_seconds(), 1),
            "config": self.config(),
            "transitions": len(self.history),
        }
