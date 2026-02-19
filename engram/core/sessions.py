"""
engram.core.sessions -- Multi-agent session management.

Provides per-client session isolation so multiple MCP clients
(Claude Code, Cursor, VS Code, etc.) can share a single Engram
server without clobbering each other's state.

Architecture:
    - ClientSession: holds per-client mutable state (source, person,
      project) that was previously stored in server.py globals.
    - SessionRegistry: thread-safe mapping of session IDs to
      ClientSession objects, with automatic creation on first access.
    - A contextvars-based current-session mechanism so synchronous
      tool functions can resolve the active session without passing
      it explicitly.

Stdio mode (single client) uses a default "stdio" session.
HTTP/SSE mode (multiple clients) creates sessions keyed by the
MCP transport session ID.
"""

from __future__ import annotations

import logging
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, Optional

log = logging.getLogger("engram.sessions")


# ---------------------------------------------------------------------------
# ClientSession — per-client mutable state
# ---------------------------------------------------------------------------


@dataclass
class ClientSession:
    """Mutable state for a single MCP client connection.

    Replaces the old global ``_current_source``, ``_current_person``,
    and ``_current_project`` variables in server.py.

    Attributes:
        session_id: Unique session identifier (transport session ID
                    or "stdio" for single-client mode).
        client_name: Human-readable client name from MCP initialize
                     handshake (e.g. "claude-code", "cursor").
        client_version: Client version string.
        source: Message source set by engram_before ("direct",
                "discord", "opencode", etc.).
        person: Resolved canonical person name from engram_before.
        project: Active project name from engram_project_init.
        created_at: Timestamp when the session was created.
        last_active: Timestamp of last tool invocation.
    """

    session_id: str = "stdio"
    client_name: str = ""
    client_version: str = ""
    source: str = "direct"
    person: str = ""
    project: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update last_active timestamp."""
        self.last_active = time.time()

    def to_dict(self) -> Dict:
        """Serialize for diagnostics / stats."""
        return {
            "session_id": self.session_id,
            "client_name": self.client_name,
            "client_version": self.client_version,
            "source": self.source,
            "person": self.person,
            "project": self.project,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "age_seconds": round(time.time() - self.created_at, 1),
        }


# ---------------------------------------------------------------------------
# SessionRegistry — thread-safe multi-session tracker
# ---------------------------------------------------------------------------

# Default session ID for stdio (single-client) mode.
DEFAULT_SESSION_ID = "stdio"

# Sessions older than this (seconds) are eligible for cleanup.
SESSION_TTL_SECONDS = 3600 * 4  # 4 hours


class SessionRegistry:
    """Thread-safe registry of active client sessions.

    In stdio mode only the default session is used.  In HTTP/SSE
    mode, each MCP transport session gets its own ClientSession.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, ClientSession] = {}
        self._lock = threading.Lock()
        # Pre-create the default session for stdio mode
        self._sessions[DEFAULT_SESSION_ID] = ClientSession(
            session_id=DEFAULT_SESSION_ID
        )

    def get(self, session_id: Optional[str] = None) -> ClientSession:
        """Get or create a ClientSession by ID.

        If *session_id* is None or empty, returns the default
        (stdio) session.  Otherwise, creates a new session on first
        access.
        """
        sid = session_id or DEFAULT_SESSION_ID
        with self._lock:
            if sid not in self._sessions:
                log.info("Creating new client session: %s", sid)
                self._sessions[sid] = ClientSession(session_id=sid)
            session = self._sessions[sid]
            session.touch()
            return session

    def remove(self, session_id: str) -> Optional[ClientSession]:
        """Remove a session (client disconnected)."""
        with self._lock:
            return self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[Dict]:
        """List all active sessions (for diagnostics)."""
        with self._lock:
            return [s.to_dict() for s in self._sessions.values()]

    def count(self) -> int:
        """Number of active sessions."""
        with self._lock:
            return len(self._sessions)

    def cleanup_stale(self, max_age: float = SESSION_TTL_SECONDS) -> int:
        """Remove sessions that have been inactive for too long.

        Returns number of sessions removed.  The default stdio
        session is never removed.
        """
        cutoff = time.time() - max_age
        removed = 0
        with self._lock:
            stale = [
                sid
                for sid, s in self._sessions.items()
                if s.last_active < cutoff and sid != DEFAULT_SESSION_ID
            ]
            for sid in stale:
                del self._sessions[sid]
                removed += 1
        if removed:
            log.info("Cleaned up %d stale sessions", removed)
        return removed


# ---------------------------------------------------------------------------
# Contextvars — resolve current session in sync tool functions
# ---------------------------------------------------------------------------

# This contextvar is set by the tool dispatch layer (or manually in
# tests) so that sync tool functions can call _get_current_session()
# to resolve the active client session without needing a Context
# parameter.
_current_session_id: ContextVar[str] = ContextVar(
    "_current_session_id", default=DEFAULT_SESSION_ID
)


def set_current_session_id(session_id: str) -> None:
    """Set the session ID for the current execution context."""
    _current_session_id.set(session_id)


def get_current_session_id() -> str:
    """Get the session ID for the current execution context."""
    return _current_session_id.get()
