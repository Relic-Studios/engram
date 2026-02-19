"""Tests for D2: Multi-agent session management.

Tests the session registry, client session isolation, and concurrent
access safety for multi-agent MCP server support.
"""

import json
import time
import threading
from unittest.mock import patch

import pytest

from engram.core.sessions import (
    ClientSession,
    SessionRegistry,
    DEFAULT_SESSION_ID,
    SESSION_TTL_SECONDS,
    get_current_session_id,
    set_current_session_id,
)
from engram.system import MemorySystem
import engram.server as server_mod


# ── ClientSession ────────────────────────────────────────────────


class TestClientSession:
    def test_default_values(self):
        s = ClientSession()
        assert s.session_id == "stdio"
        assert s.source == "direct"
        assert s.person == ""
        assert s.project == ""
        assert s.client_name == ""
        assert s.client_version == ""

    def test_custom_values(self):
        s = ClientSession(
            session_id="abc123",
            client_name="cursor",
            client_version="0.42",
            source="ide",
            person="alice",
            project="my-app",
        )
        assert s.session_id == "abc123"
        assert s.client_name == "cursor"
        assert s.person == "alice"
        assert s.project == "my-app"

    def test_touch_updates_last_active(self):
        s = ClientSession()
        old = s.last_active
        time.sleep(0.01)
        s.touch()
        assert s.last_active > old

    def test_to_dict(self):
        s = ClientSession(session_id="test", client_name="vscode")
        d = s.to_dict()
        assert d["session_id"] == "test"
        assert d["client_name"] == "vscode"
        assert "age_seconds" in d
        assert d["age_seconds"] >= 0

    def test_mutable_state(self):
        s = ClientSession()
        s.source = "discord"
        s.person = "bob"
        s.project = "engram"
        assert s.source == "discord"
        assert s.person == "bob"
        assert s.project == "engram"


# ── SessionRegistry ─────────────────────────────────────────────


class TestSessionRegistry:
    def test_default_session_exists(self):
        reg = SessionRegistry()
        s = reg.get()
        assert s.session_id == DEFAULT_SESSION_ID

    def test_get_creates_new_session(self):
        reg = SessionRegistry()
        s = reg.get("client-1")
        assert s.session_id == "client-1"
        assert reg.count() == 2  # default + client-1

    def test_get_returns_same_session(self):
        reg = SessionRegistry()
        s1 = reg.get("client-1")
        s1.person = "alice"
        s2 = reg.get("client-1")
        assert s2.person == "alice"
        assert s1 is s2

    def test_none_returns_default(self):
        reg = SessionRegistry()
        s = reg.get(None)
        assert s.session_id == DEFAULT_SESSION_ID

    def test_empty_string_returns_default(self):
        reg = SessionRegistry()
        s = reg.get("")
        assert s.session_id == DEFAULT_SESSION_ID

    def test_multiple_sessions_isolated(self):
        reg = SessionRegistry()
        s1 = reg.get("cursor")
        s2 = reg.get("vscode")
        s1.person = "alice"
        s1.project = "project-a"
        s2.person = "bob"
        s2.project = "project-b"
        assert reg.get("cursor").person == "alice"
        assert reg.get("vscode").person == "bob"
        assert reg.get("cursor").project == "project-a"
        assert reg.get("vscode").project == "project-b"

    def test_remove(self):
        reg = SessionRegistry()
        reg.get("temp")
        assert reg.count() == 2
        removed = reg.remove("temp")
        assert removed is not None
        assert removed.session_id == "temp"
        assert reg.count() == 1

    def test_remove_nonexistent(self):
        reg = SessionRegistry()
        assert reg.remove("nope") is None

    def test_list_sessions(self):
        reg = SessionRegistry()
        reg.get("a")
        reg.get("b")
        sessions = reg.list_sessions()
        ids = {s["session_id"] for s in sessions}
        assert DEFAULT_SESSION_ID in ids
        assert "a" in ids
        assert "b" in ids

    def test_cleanup_stale(self):
        reg = SessionRegistry()
        s = reg.get("old-session")
        # Backdate the session
        s.last_active = time.time() - SESSION_TTL_SECONDS - 100
        removed = reg.cleanup_stale()
        assert removed == 1
        assert reg.count() == 1  # only default remains

    def test_cleanup_preserves_default(self):
        reg = SessionRegistry()
        default = reg.get()
        default.last_active = time.time() - SESSION_TTL_SECONDS - 100
        removed = reg.cleanup_stale()
        assert removed == 0
        assert reg.count() == 1

    def test_thread_safety(self):
        """Concurrent get/remove from multiple threads."""
        reg = SessionRegistry()
        errors = []

        def worker(i):
            try:
                sid = f"thread-{i}"
                s = reg.get(sid)
                s.person = f"person-{i}"
                time.sleep(0.001)
                assert reg.get(sid).person == f"person-{i}"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert reg.count() == 21  # default + 20 threads


# ── Contextvars ──────────────────────────────────────────────────


class TestContextVars:
    def test_default_session_id(self):
        assert get_current_session_id() == DEFAULT_SESSION_ID

    def test_set_and_get(self):
        set_current_session_id("test-session")
        assert get_current_session_id() == "test-session"
        # Reset
        set_current_session_id(DEFAULT_SESSION_ID)

    def test_thread_isolation(self):
        """Each thread gets its own contextvar value."""
        results = {}

        def worker(i):
            set_current_session_id(f"thread-{i}")
            time.sleep(0.01)
            results[i] = get_current_session_id()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i in range(5):
            assert results[i] == f"thread-{i}"


# ── Server Integration ───────────────────────────────────────────


@pytest.fixture
def server_system(config_with_soul):
    """Wire up a real MemorySystem for server integration tests."""
    cfg = config_with_soul
    system = MemorySystem(config=cfg)

    old_system = server_mod._system
    old_source = server_mod._current_source
    old_person = server_mod._current_person
    old_project = server_mod._current_project
    old_sessions = server_mod._sessions
    server_mod._system = system
    server_mod._current_source = "direct"
    server_mod._current_person = "tester"
    server_mod._current_project = ""
    server_mod._sessions = SessionRegistry()

    yield system

    server_mod._system = old_system
    server_mod._current_source = old_source
    server_mod._current_person = old_person
    server_mod._current_project = old_project
    server_mod._sessions = old_sessions
    system.close()


class TestServerSessionIntegration:
    """Verify that server tools use the session registry."""

    def test_engram_before_updates_session(self, server_system):
        """engram_before should update the session's source and person."""
        server_mod.engram_before(person="alice", message="hello", source="ide")
        session = server_mod._get_session()
        assert session.source == "ide"
        assert session.person == "alice"

    def test_engram_before_updates_legacy_globals(self, server_system):
        """engram_before should also update legacy globals."""
        server_mod.engram_before(person="alice", message="hello", source="test_source")
        assert server_mod._current_source == "test_source"
        assert server_mod._current_person == "alice"

    def test_engram_project_init_updates_session(self, server_system):
        """engram_project_init should update the session's project."""
        server_mod.engram_project_init(project_name="my-app")
        session = server_mod._get_session()
        assert session.project == "my-app"

    def test_engram_sessions_tool(self, server_system):
        """engram_sessions should return session list."""
        result = server_mod.engram_sessions()
        data = json.loads(result)
        assert "active_sessions" in data
        assert data["count"] >= 1

    def test_engram_register_client(self, server_system):
        """engram_register_client should create a named session."""
        result = server_mod.engram_register_client(
            client_name="cursor", client_version="0.42", session_id="test-123"
        )
        data = json.loads(result)
        assert data["client_name"] == "cursor"
        assert data["client_version"] == "0.42"
        assert data["session_id"] == "test-123"

    def test_session_isolation_via_register(self, server_system):
        """Two registered clients should have isolated state."""
        # Register two clients
        server_mod.engram_register_client(client_name="cursor", session_id="client-a")
        set_current_session_id("client-a")
        server_mod.engram_before(person="alice", message="hi", source="cursor")

        server_mod.engram_register_client(client_name="vscode", session_id="client-b")
        set_current_session_id("client-b")
        server_mod.engram_before(person="bob", message="hi", source="vscode")

        # Verify isolation
        session_a = server_mod._sessions.get("client-a")
        session_b = server_mod._sessions.get("client-b")
        assert session_a.person == "alice"
        assert session_a.source == "cursor"
        assert session_b.person == "bob"
        assert session_b.source == "vscode"


# ── Concurrent Database Access ───────────────────────────────────


class TestConcurrentDBAccess:
    """Verify that concurrent reads/writes don't crash."""

    def test_concurrent_trace_reads(self, server_system):
        """Multiple threads reading traces simultaneously."""
        ep = server_system.episodic
        # Seed some data
        for i in range(5):
            ep.log_trace(
                kind="episode",
                content=f"concurrent read test {i}",
                tags=["test"],
                salience=0.5,
            )

        errors = []

        def reader():
            try:
                ep.get_by_salience(limit=5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_concurrent_writes(self, server_system):
        """Multiple threads writing traces simultaneously."""
        ep = server_system.episodic
        errors = []

        def writer(i):
            try:
                ep.log_trace(
                    kind="episode",
                    content=f"concurrent write test {i}",
                    tags=["concurrent"],
                    salience=0.5,
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_mixed_reads_and_writes(self, server_system):
        """Concurrent reads and writes should not deadlock."""
        ep = server_system.episodic
        # Seed initial data
        for i in range(3):
            ep.log_trace(
                kind="episode",
                content=f"mixed test seed {i}",
                tags=["test"],
                salience=0.5,
            )

        errors = []

        def reader():
            try:
                for _ in range(5):
                    ep.get_by_salience(limit=5)
            except Exception as e:
                errors.append(e)

        def writer(i):
            try:
                for j in range(3):
                    ep.log_trace(
                        kind="episode",
                        content=f"mixed test write {i}-{j}",
                        tags=["mixed"],
                        salience=0.5,
                    )
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=reader))
            threads.append(threading.Thread(target=writer, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
