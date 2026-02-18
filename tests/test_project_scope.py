"""Tests for project-scoped memory isolation.

Verifies that traces, messages, and sessions can be scoped to
specific projects, and that queries correctly filter by project
while including global (unscoped) data.
"""

import pytest

from engram.episodic.store import EpisodicStore


@pytest.fixture
def store(config):
    """Fresh EpisodicStore for project scope tests."""
    ep = EpisodicStore(config.db_path)
    yield ep
    ep.close()


# ---------------------------------------------------------------------------
# Schema: project column exists
# ---------------------------------------------------------------------------


class TestProjectColumn:
    """Verify project column exists on all tables."""

    def test_traces_has_project(self, store):
        """Traces table should have a project column."""
        tid = store.log_trace(
            content="Test trace",
            kind="episode",
            tags=["test"],
            project="my-project",
        )
        trace = store.get_trace(tid)
        assert trace is not None
        assert trace.get("project") == "my-project"

    def test_traces_default_project_empty(self, store):
        """Default project should be empty string (global)."""
        tid = store.log_trace(
            content="Global trace",
            kind="episode",
            tags=["test"],
        )
        trace = store.get_trace(tid)
        assert trace.get("project") == ""

    def test_messages_has_project(self, store):
        """Messages table should have a project column."""
        mid = store.log_message(
            person="alice",
            speaker="alice",
            content="Hello from project",
            source="test",
            project="my-project",
        )
        msg = store.get_message(mid)
        assert msg is not None
        assert msg.get("project") == "my-project"

    def test_sessions_has_project(self, store):
        """Sessions table should have a project column."""
        sid = store.start_session(person="alice", project="my-project")
        sessions = store.get_recent_sessions(person="alice", limit=1)
        assert len(sessions) >= 1
        assert sessions[0].get("project") == "my-project"

    def test_events_has_project(self, store):
        """Events table should have a project column."""
        eid = store.log_event(
            type="test_event",
            description="A test event",
            project="my-project",
        )
        # Events don't have a direct get_event, but the insert succeeds
        assert eid  # non-empty string


# ---------------------------------------------------------------------------
# Project-scoped queries
# ---------------------------------------------------------------------------


class TestProjectScopedQueries:
    """Verify queries filter by project correctly."""

    def test_get_by_salience_project_filter(self, store):
        """get_by_salience should return only project + global traces."""
        # Create traces in different projects
        store.log_trace(
            content="Alpha trace",
            kind="episode",
            tags=["a"],
            salience=0.9,
            project="alpha",
        )
        store.log_trace(
            content="Beta trace",
            kind="episode",
            tags=["b"],
            salience=0.9,
            project="beta",
        )
        store.log_trace(
            content="Global trace", kind="episode", tags=["g"], salience=0.9
        )

        # Query for alpha project — should get alpha + global, not beta
        results = store.get_by_salience(project="alpha", limit=10)
        contents = [r["content"] for r in results]
        assert "Alpha trace" in contents
        assert "Global trace" in contents
        assert "Beta trace" not in contents

    def test_get_by_salience_no_project_filter(self, store):
        """get_by_salience without project returns all traces."""
        store.log_trace(
            content="Alpha trace",
            kind="episode",
            tags=["a"],
            salience=0.9,
            project="alpha",
        )
        store.log_trace(
            content="Beta trace",
            kind="episode",
            tags=["b"],
            salience=0.9,
            project="beta",
        )

        results = store.get_by_salience(limit=10)
        contents = [r["content"] for r in results]
        assert "Alpha trace" in contents
        assert "Beta trace" in contents

    def test_get_traces_project_filter(self, store):
        """get_traces should filter by project."""
        store.log_trace(
            content="Code pattern A",
            kind="code_pattern",
            tags=[],
            salience=0.8,
            project="project-a",
        )
        store.log_trace(
            content="Code pattern B",
            kind="code_pattern",
            tags=[],
            salience=0.8,
            project="project-b",
        )
        store.log_trace(
            content="Global pattern", kind="code_pattern", tags=[], salience=0.8
        )

        results = store.get_traces(kind="code_pattern", project="project-a")
        contents = [r["content"] for r in results]
        assert "Code pattern A" in contents
        assert "Global pattern" in contents
        assert "Code pattern B" not in contents

    def test_get_traces_by_kind_project_filter(self, store):
        """get_traces_by_kind should filter by project."""
        store.log_trace(
            content="ADR for alpha",
            kind="architecture_decision",
            tags=[],
            salience=0.95,
            project="alpha",
        )
        store.log_trace(
            content="ADR for beta",
            kind="architecture_decision",
            tags=[],
            salience=0.95,
            project="beta",
        )

        results = store.get_traces_by_kind("architecture_decision", project="alpha")
        contents = [r["content"] for r in results]
        assert "ADR for alpha" in contents
        assert "ADR for beta" not in contents


# ---------------------------------------------------------------------------
# Isolation correctness
# ---------------------------------------------------------------------------


class TestProjectIsolation:
    """Verify that project scoping provides proper isolation."""

    def test_different_projects_fully_isolated(self, store):
        """Traces in project A should not appear in project B queries."""
        for i in range(5):
            store.log_trace(
                content=f"Alpha-{i}",
                kind="episode",
                tags=[],
                salience=0.5,
                project="alpha",
            )
            store.log_trace(
                content=f"Beta-{i}",
                kind="episode",
                tags=[],
                salience=0.5,
                project="beta",
            )

        alpha_traces = store.get_by_salience(project="alpha", limit=50)
        alpha_contents = {r["content"] for r in alpha_traces}

        # No beta traces in alpha results
        for content in alpha_contents:
            assert not content.startswith("Beta-")

    def test_global_traces_visible_in_all_projects(self, store):
        """Global traces (project='') should appear in all project queries."""
        store.log_trace(
            content="Universal knowledge",
            kind="factual",
            tags=[],
            salience=0.9,
            project="",
        )

        for proj in ("alpha", "beta", "gamma"):
            results = store.get_by_salience(project=proj, limit=10)
            contents = [r["content"] for r in results]
            assert "Universal knowledge" in contents, (
                f"Global trace missing from project '{proj}'"
            )
