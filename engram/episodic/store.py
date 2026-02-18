"""
Episodic memory store — SQLite-backed storage for conversations,
events, and experiential traces.

Every interaction leaves a trace. Salience determines what survives.
"""

import sqlite3
import json
import math
import uuid
from pathlib import Path
from typing import Any, Callable, List, Optional, Dict
from datetime import datetime, timedelta, timezone


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EpisodicStore:
    """SQLite-based episodic memory: messages, traces, and events."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

        # Callback fired after every log_trace().  Signature:
        #   (trace_id: str, content: str, metadata: dict) -> None
        # Used by MemorySystem to push new traces into ChromaDB for
        # incremental vector indexing (FR-2).
        self._on_trace_logged: Optional[Callable] = None

        # Batch write support: when _batch_depth > 0, individual
        # commit() calls are suppressed and a single commit runs
        # when the outermost batch context exits.
        self._batch_depth: int = 0

    # ── Batch writes ────────────────────────────────────────────

    class _BatchContext:
        """Context manager that defers commits until exit."""

        def __init__(self, store: "EpisodicStore") -> None:
            self._store = store

        def __enter__(self) -> "EpisodicStore":
            self._store._batch_depth += 1
            return self._store

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            self._store._batch_depth -= 1
            if self._store._batch_depth <= 0:
                self._store._batch_depth = 0
                if exc_type is None:
                    self._store.conn.commit()
                else:
                    # Rollback on exception
                    try:
                        self._store.conn.rollback()
                    except Exception:
                        pass

    def batch(self) -> "_BatchContext":
        """Return a context manager that batches SQLite writes.

        Within the ``batch()`` block, individual ``conn.commit()``
        calls in ``log_message()``, ``log_trace()``, ``reinforce()``,
        etc. are suppressed.  A single commit runs when the block
        exits successfully, or a rollback on exception.

        Under WAL mode this reduces disk I/O by 5-10x for the
        after-pipeline which typically does 3-5 writes per call.

        Usage::

            with episodic.batch():
                episodic.log_message(...)
                episodic.log_trace(...)
                episodic.reinforce(...)
            # single commit happens here
        """
        return self._BatchContext(self)

    def _commit(self) -> None:
        """Commit unless inside a batch context."""
        if self._batch_depth <= 0:
            self.conn.commit()

    # ── Schema ────────────────────────────────────────────────

    def _create_tables(self):
        c = self.conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              TEXT PRIMARY KEY,
                person          TEXT,
                speaker         TEXT,
                content         TEXT,
                source          TEXT,
                timestamp       TEXT,
                salience        REAL DEFAULT 0.5,
                signal          JSON,
                metadata        JSON,
                access_count    INTEGER DEFAULT 0,
                last_accessed   TEXT,
                project         TEXT DEFAULT ''
            )
        """)

        # Migration: add access_count/last_accessed to existing databases
        # that were created before these columns existed.
        for col, col_def in [
            ("access_count", "INTEGER DEFAULT 0"),
            ("last_accessed", "TEXT"),
        ]:
            try:
                c.execute(f"ALTER TABLE messages ADD COLUMN {col} {col_def}")
            except sqlite3.OperationalError:
                pass  # column already exists

        c.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id              TEXT PRIMARY KEY,
                content         TEXT,
                created         TEXT,
                kind            TEXT,
                tags            JSON,
                salience        REAL DEFAULT 0.5,
                tokens          INTEGER,
                access_count    INTEGER DEFAULT 0,
                last_accessed   TEXT,
                metadata        JSON,
                project         TEXT DEFAULT ''
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id          TEXT PRIMARY KEY,
                type        TEXT,
                description TEXT,
                person      TEXT,
                timestamp   TEXT,
                salience    REAL DEFAULT 0.5,
                metadata    JSON,
                project     TEXT DEFAULT ''
            )
        """)

        # Indexes (for messages, traces, events — created above)
        for stmt in [
            "CREATE INDEX IF NOT EXISTS idx_messages_person    ON messages(person)",
            "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_messages_salience  ON messages(salience)",
            "CREATE INDEX IF NOT EXISTS idx_traces_salience    ON traces(salience)",
            "CREATE INDEX IF NOT EXISTS idx_traces_kind        ON traces(kind)",
            "CREATE INDEX IF NOT EXISTS idx_events_type        ON events(type)",
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp   ON events(timestamp)",
        ]:
            c.execute(stmt)

        # FTS5 full-text search with Porter stemming.
        #
        # Porter stemming enables morphological matching: "running"
        # matches "run", "cats" matches "cat", etc.  The `porter`
        # tokenizer wraps `unicode61` (the default) and applies the
        # Porter stemming algorithm to each token.
        #
        # Migration: if an old FTS table exists without Porter stemming,
        # we drop and recreate it so the index uses the new tokenizer.
        # The sync triggers will repopulate data on next insert/update.
        _FTS_TOKENIZER = "tokenize='porter unicode61'"
        for table, fts in [("messages", "messages_fts"), ("traces", "traces_fts")]:
            # Check if FTS table exists and needs migration
            existing = c.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (fts,),
            ).fetchone()
            if existing:
                sql = existing[0] or ""
                if "porter" not in sql.lower():
                    # Old FTS table without stemming — drop and recreate
                    c.execute(f"DROP TABLE IF EXISTS {fts}")
                    # Also drop triggers so they can be recreated
                    for suffix in ("ai", "ad", "au"):
                        c.execute(f"DROP TRIGGER IF EXISTS {table}_{suffix}")

            c.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {fts}
                USING fts5(content, content={table}, content_rowid=rowid,
                           {_FTS_TOKENIZER})
            """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                person      TEXT,
                started     TEXT,
                ended       TEXT,
                message_count INTEGER DEFAULT 0,
                summary     TEXT,
                metadata    JSON,
                project     TEXT DEFAULT ''
            )
        """)

        c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_person  ON sessions(person)")
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started)"
        )

        # Migration (v2 pivot): add project column to existing databases
        # that were created before the column existed.  Must run AFTER
        # all CREATE TABLE statements so the tables exist.
        for table in ("messages", "traces", "events", "sessions"):
            try:
                c.execute(f"ALTER TABLE {table} ADD COLUMN project TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # column already exists

        # Project indexes (must be after migration adds the column)
        for stmt in [
            "CREATE INDEX IF NOT EXISTS idx_traces_project   ON traces(project)",
            "CREATE INDEX IF NOT EXISTS idx_messages_project  ON messages(project)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_project  ON sessions(project)",
        ]:
            c.execute(stmt)

        # Relationship graph — lightweight temporal knowledge graph
        # (Zep/Graphiti-inspired, no Neo4j dependency).
        c.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id              TEXT PRIMARY KEY,
                subject         TEXT NOT NULL,
                predicate       TEXT NOT NULL,
                object          TEXT NOT NULL,
                valid_from      TEXT,
                valid_until     TEXT,
                confidence      REAL DEFAULT 1.0,
                source_trace_id TEXT,
                metadata        JSON
            )
        """)
        for stmt in [
            "CREATE INDEX IF NOT EXISTS idx_rel_subject   ON relationships(subject)",
            "CREATE INDEX IF NOT EXISTS idx_rel_object    ON relationships(object)",
            "CREATE INDEX IF NOT EXISTS idx_rel_predicate ON relationships(predicate)",
            "CREATE INDEX IF NOT EXISTS idx_rel_valid     ON relationships(valid_until)",
        ]:
            c.execute(stmt)

        # Triggers to keep FTS in sync with source tables
        for table in ("messages", "traces"):
            fts = f"{table}_fts"
            c.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {table}_ai AFTER INSERT ON {table}
                BEGIN
                    INSERT INTO {fts}(rowid, content)
                    VALUES (new.rowid, new.content);
                END
            """)
            c.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {table}_ad AFTER DELETE ON {table}
                BEGIN
                    INSERT INTO {fts}({fts}, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                END
            """)
            c.execute(f"""
                CREATE TRIGGER IF NOT EXISTS {table}_au AFTER UPDATE ON {table}
                BEGIN
                    INSERT INTO {fts}({fts}, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                    INSERT INTO {fts}(rowid, content)
                    VALUES (new.rowid, new.content);
                END
            """)

        # Rebuild FTS indexes to populate from source tables.
        # This is a no-op if the tables are already in sync, and
        # is necessary after a migration (drop + recreate).
        for fts in ("messages_fts", "traces_fts"):
            try:
                c.execute(f"INSERT INTO {fts}({fts}) VALUES('rebuild')")
            except sqlite3.OperationalError:
                pass  # table might be empty, that's fine

        # Apply symbol expansion to the FTS index.
        # After the trigger-based rebuild populates raw content,
        # re-index with expanded content for code symbol search.
        self._rebuild_fts_expanded(c)

        self.conn.commit()

    # ── FTS symbol expansion ─────────────────────────────────

    def _fts_replace_expanded(self, table: str, rowid: int, content: str) -> None:
        """Replace a raw FTS entry with symbol-expanded content.

        Called after INSERT (which triggers raw-content FTS insert).
        Replaces the trigger's entry with expanded content that splits
        compound identifiers (camelCase, snake_case, dot.paths) into
        individual searchable tokens.

        This is a no-op if the tokenizer module is not available or
        the content has no compound identifiers to expand.
        """
        try:
            from engram.search.tokenizer import expand_text
        except ImportError:
            return  # tokenizer not available — keep raw content

        expanded = expand_text(content)
        if expanded == content:
            return  # nothing to expand

        fts = f"{table}_fts"
        try:
            # Delete the trigger-inserted raw entry, insert expanded
            self.conn.execute(
                f"INSERT INTO {fts}({fts}, rowid, content) VALUES('delete', ?, ?)",
                (rowid, content),
            )
            self.conn.execute(
                f"INSERT INTO {fts}(rowid, content) VALUES(?, ?)",
                (rowid, expanded),
            )
        except Exception:
            pass  # best-effort — don't break the write path

    def _get_last_rowid(self) -> int:
        """Get the rowid of the last INSERT."""
        row = self.conn.execute("SELECT last_insert_rowid()").fetchone()
        return row[0] if row else 0

    def _rebuild_fts_expanded(self, cursor=None) -> int:
        """Re-expand all FTS entries with symbol-tokenized content.

        Called during table creation / migration to ensure existing
        content gets symbol expansion applied.  Returns the number
        of entries expanded.
        """
        try:
            from engram.search.tokenizer import expand_text
        except ImportError:
            return 0

        c = cursor or self.conn
        expanded_count = 0

        for table, fts in [("messages", "messages_fts"), ("traces", "traces_fts")]:
            try:
                rows = c.execute(f"SELECT rowid, content FROM {table}").fetchall()
            except sqlite3.OperationalError:
                continue

            for row in rows:
                rowid = row[0]
                content = row[1] or ""
                expanded = expand_text(content)
                if expanded != content:
                    try:
                        c.execute(
                            f"INSERT INTO {fts}({fts}, rowid, content) "
                            f"VALUES('delete', ?, ?)",
                            (rowid, content),
                        )
                        c.execute(
                            f"INSERT INTO {fts}(rowid, content) VALUES(?, ?)",
                            (rowid, expanded),
                        )
                        expanded_count += 1
                    except Exception:
                        pass

        return expanded_count

    # ── Write ─────────────────────────────────────────────────

    # Idempotency window: reject duplicate messages with same person,
    # speaker, and content logged within this many seconds.
    DEDUP_WINDOW_SECONDS: int = 30

    def log_message(
        self,
        person: str,
        speaker: str,
        content: str,
        source: str,
        salience: float = 0.5,
        signal: Optional[Dict] = None,
        project: str = "",
        **metadata,
    ) -> str:
        """Record a conversation message. Returns the message ID.

        Idempotency: if an identical message (same person, speaker,
        content) was logged within the last ``DEDUP_WINDOW_SECONDS``,
        the existing message ID is returned instead of creating a
        duplicate.  This guards against Discord firing messageCreate
        twice or engram_after being called multiple times.
        """
        # -- Idempotency check: skip if duplicate within window ------------
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self.DEDUP_WINDOW_SECONDS
        )
        cutoff_iso = cutoff.isoformat().replace("+00:00", "Z")
        existing = self.conn.execute(
            """
            SELECT id FROM messages
            WHERE person = ? AND speaker = ? AND content = ?
              AND timestamp >= ?
            LIMIT 1
            """,
            (person, speaker, content, cutoff_iso),
        ).fetchone()
        if existing:
            return existing[0]  # return existing ID, no duplicate

        msg_id = _generate_id()
        self.conn.execute(
            """
            INSERT INTO messages (id, person, speaker, content, source,
                                  timestamp, salience, signal, metadata,
                                  project)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                msg_id,
                person,
                speaker,
                content,
                source,
                _now(),
                salience,
                json.dumps(signal) if signal else None,
                json.dumps(metadata) if metadata else None,
                project,
            ),
        )
        # Replace raw FTS entry with symbol-expanded content
        rowid = self._get_last_rowid()
        self._fts_replace_expanded("messages", rowid, content)
        self._commit()
        return msg_id

    def log_trace(
        self,
        content: str,
        kind: str,
        tags: List[str],
        salience: float = 0.5,
        project: str = "",
        **metadata,
    ) -> str:
        """Record an experiential trace (summary, insight, reflection). Returns trace ID.

        Idempotency: if a trace with identical content and kind was
        created within the last ``DEDUP_WINDOW_SECONDS``, the existing
        trace ID is returned instead of creating a duplicate.
        """
        from engram.core.types import TRACE_KINDS

        if kind not in TRACE_KINDS:
            raise ValueError(
                f"Invalid trace kind {kind!r}. Must be one of {sorted(TRACE_KINDS)}"
            )

        # -- Idempotency check: skip if duplicate within window ------------
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self.DEDUP_WINDOW_SECONDS
        )
        cutoff_iso = cutoff.isoformat().replace("+00:00", "Z")
        existing = self.conn.execute(
            """
            SELECT id FROM traces
            WHERE content = ? AND kind = ? AND created >= ?
            LIMIT 1
            """,
            (content, kind, cutoff_iso),
        ).fetchone()
        if existing:
            return existing[0]  # return existing ID, no duplicate

        trace_id = _generate_id()
        now = _now()
        from engram.core.tokens import estimate_tokens

        tokens = estimate_tokens(content)
        self.conn.execute(
            """
            INSERT INTO traces (id, content, created, kind, tags, salience,
                                tokens, access_count, last_accessed, metadata,
                                project)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                trace_id,
                content,
                now,
                kind,
                json.dumps(tags),
                salience,
                tokens,
                now,
                json.dumps(metadata) if metadata else None,
                project,
            ),
        )
        # Replace raw FTS entry with symbol-expanded content
        rowid = self._get_last_rowid()
        self._fts_replace_expanded("traces", rowid, content)
        self._commit()

        # Fire incremental vector-indexing callback (FR-2)
        if self._on_trace_logged is not None:
            try:
                trace_meta = dict(metadata) if metadata else {}
                trace_meta.update(
                    {
                        "kind": kind,
                        "salience": salience,
                        "created": now,
                        "project": project,
                    }
                )
                self._on_trace_logged(trace_id, content, trace_meta)
            except Exception:
                pass  # best-effort; don't break trace logging

        return trace_id

    def log_event(
        self,
        type: str,
        description: str,
        person: Optional[str] = None,
        salience: float = 0.5,
        project: str = "",
        **metadata,
    ) -> str:
        """Record a discrete event. Returns event ID."""
        event_id = _generate_id()
        self.conn.execute(
            """
            INSERT INTO events (id, type, description, person,
                                timestamp, salience, metadata, project)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                type,
                description,
                person,
                _now(),
                salience,
                json.dumps(metadata) if metadata else None,
                project,
            ),
        )
        self._commit()
        return event_id

    # ── Read ──────────────────────────────────────────────────

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        """Convert a sqlite3.Row to a plain dict, deserializing JSON fields."""
        d = dict(row)
        for key in ("signal", "metadata", "tags"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def get_messages(
        self,
        person: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 50,
        min_salience: float = 0.0,
    ) -> List[Dict]:
        """Retrieve messages with optional filters."""
        clauses = ["salience >= ?"]
        params: list = [min_salience]

        if person:
            clauses.append("person = ?")
            params.append(person)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)

        where = " AND ".join(clauses)
        params.append(limit)

        rows = self.conn.execute(
            f"SELECT * FROM messages WHERE {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_traces(
        self,
        tags: Optional[List[str]] = None,
        kind: Optional[str] = None,
        min_salience: float = 0.0,
        limit: int = 50,
        project: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve traces with optional filters.

        When ``project`` is provided, only traces scoped to that project
        (or global traces with project='') are returned.
        """
        clauses = ["salience >= ?"]
        params: list = [min_salience]

        if kind:
            clauses.append("kind = ?")
            params.append(kind)

        if project is not None:
            clauses.append("(project = ? OR project = '')")
            params.append(project)

        where = " AND ".join(clauses)
        params.append(limit)

        rows = self.conn.execute(
            f"SELECT * FROM traces WHERE {where} ORDER BY salience DESC LIMIT ?",
            params,
        ).fetchall()

        results = [self._row_to_dict(r) for r in rows]

        # Tag filtering in Python — SQLite json_each works but is fragile
        # across versions; this is cleaner for small result sets.
        if tags:
            tag_set = set(t.lower() for t in tags)
            results = [
                r
                for r in results
                if r.get("tags") and tag_set.intersection(t.lower() for t in r["tags"])
            ]

        return results

    def get_recent_messages(self, person: str, limit: int = 20) -> List[Dict]:
        """Get the most recent *unarchived* messages for a person, in chronological order.

        Archived messages (from compaction) are excluded so that only
        live conversation history is returned.  Without this filter,
        compacted messages would crowd out recent context.

        Deduplicates consecutive messages with identical speaker+content
        (keeps the earliest of each run).  This cleans up any duplicates
        that were logged before the idempotency guard was added.
        """
        # Over-fetch to account for duplicates that will be removed
        rows = self.conn.execute(
            """
            SELECT * FROM messages
            WHERE person = ?
              AND COALESCE(json_extract(metadata, '$.archived'), 0) = 0
            ORDER BY timestamp DESC, rowid DESC
            LIMIT ?
            """,
            (person, limit * 2),
        ).fetchall()
        results = [self._row_to_dict(r) for r in rows]
        results.reverse()  # chronological

        # Deduplicate: remove consecutive identical speaker+content pairs
        deduped: List[Dict] = []
        for msg in results:
            key = (msg.get("speaker", ""), msg.get("content", ""))
            if (
                deduped
                and (deduped[-1].get("speaker", ""), deduped[-1].get("content", ""))
                == key
            ):
                continue  # skip duplicate
            deduped.append(msg)

        return deduped[-limit:]  # trim to requested limit

    def get_by_salience(
        self,
        person: Optional[str] = None,
        limit: int = 30,
        project: Optional[str] = None,
    ) -> List[Dict]:
        """Get the highest-salience traces, optionally filtered by person tag and project."""
        if project is not None:
            rows = self.conn.execute(
                "SELECT * FROM traces WHERE (project = ? OR project = '') "
                "ORDER BY salience DESC LIMIT ?",
                (project, limit * 3),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM traces ORDER BY salience DESC LIMIT ?",
                (limit * 3,),  # over-fetch to allow Python tag filtering
            ).fetchall()

        results = [self._row_to_dict(r) for r in rows]

        if person:
            person_lower = person.lower()
            results = [
                r
                for r in results
                if r.get("tags") and person_lower in [t.lower() for t in r["tags"]]
            ]

        return results[:limit]

    def search_messages(
        self, query: str, person: Optional[str] = None, limit: int = 20
    ) -> List[Dict]:
        """Full-text search over message content."""
        if person:
            rows = self.conn.execute(
                """
                SELECT m.* FROM messages m
                JOIN messages_fts fts ON m.rowid = fts.rowid
                WHERE messages_fts MATCH ? AND m.person = ?
                ORDER BY fts.rank
                LIMIT ?
                """,
                (query, person, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT m.* FROM messages m
                JOIN messages_fts fts ON m.rowid = fts.rowid
                WHERE messages_fts MATCH ?
                ORDER BY fts.rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def search_traces(self, query: str, limit: int = 20) -> List[Dict]:
        """Full-text search over trace content."""
        rows = self.conn.execute(
            """
            SELECT t.* FROM traces t
            JOIN traces_fts fts ON t.rowid = fts.rowid
            WHERE traces_fts MATCH ?
            ORDER BY fts.rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ── Single lookups ────────────────────────────────────────

    def get_trace(self, id: str) -> Optional[Dict]:
        row = self.conn.execute("SELECT * FROM traces WHERE id = ?", (id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def get_message(self, id: str) -> Optional[Dict]:
        row = self.conn.execute("SELECT * FROM messages WHERE id = ?", (id,)).fetchone()
        return self._row_to_dict(row) if row else None

    # ── Salience management ───────────────────────────────────

    def reinforce(self, table: str, id: str, delta: float):
        """Increase salience for a record. Clamps to [0, 1]."""
        self._validate_table(table)
        self.conn.execute(
            f"UPDATE {table} SET salience = MIN(1.0, salience + ?) WHERE id = ?",
            (abs(delta), id),
        )
        self._commit()

    def weaken(self, table: str, id: str, delta: float):
        """Decrease salience for a record. Clamps to [0, 1]."""
        self._validate_table(table)
        self.conn.execute(
            f"UPDATE {table} SET salience = MAX(0.0, salience - ?) WHERE id = ?",
            (abs(delta), id),
        )
        self._commit()

    def update_access(self, table: str, id: str):
        """Record that a trace or message was accessed (retrieved for context).

        Increments ``access_count`` and updates ``last_accessed`` for both
        traces and messages.  Access tracking enables decay resistance
        (frequently-retrieved items decay slower) and retrieval analytics.
        """
        self._validate_table(table)
        if table == "events":
            return  # events don't track access
        now = _now()
        self.conn.execute(
            f"UPDATE {table} SET access_count = COALESCE(access_count, 0) + 1, "
            f"last_accessed = ? WHERE id = ?",
            (now, id),
        )
        self._commit()

    # ── Statistics ─────────────────────────────────────────────

    def count_messages(self, person: Optional[str] = None) -> int:
        if person:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM messages WHERE person = ?", (person,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()
        return row[0] if row else 0

    def count_traces(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM traces").fetchone()
        return row[0] if row else 0

    def count_events(self, type: Optional[str] = None) -> int:
        """Count events, optionally filtered by type."""
        if type:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM events WHERE type = ?", (type,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return row[0] if row else 0

    def get_events(
        self,
        type: Optional[str] = None,
        person: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Retrieve events, optionally filtered by type and/or person."""
        clauses: List[str] = []
        params: list = []
        if type:
            clauses.append("type = ?")
            params.append(type)
        if person:
            clauses.append("person = ?")
            params.append(person)

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        params.append(limit)

        rows = self.conn.execute(
            f"SELECT * FROM events{where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def avg_salience(self, table: str) -> float:
        self._validate_table(table)
        row = self.conn.execute(f"SELECT AVG(salience) FROM {table}").fetchone()
        return round(row[0], 4) if row and row[0] is not None else 0.0

    # ── Decay & pruning ───────────────────────────────────────

    # Half-life for access recency weighting (30 days in hours).
    # Old accesses contribute less to decay resistance than recent ones.
    ACCESS_RECENCY_HALF_LIFE_HOURS: float = 720.0  # 30 days

    def decay_pass(self, half_life_hours: float, coherence: float):
        """
        Run adaptive exponential decay across all traces.

        The decay rate is derived from the half-life but modulated by
        coherence: high coherence (system is stable, well-integrated)
        means memories can decay faster — we don't need as many.  Low
        coherence means hold on to more.

        Access count provides resistance: frequently-retrieved traces
        decay slower because they're clearly useful.  The access count
        is modulated by **recency** (ACT-R-inspired): old accesses
        contribute less to resistance than recent ones.

        Formula:
            decay_rate = ln(2) / half_life * coherence_factor
            coherence_factor = 0.5 + coherence  (range ~0.5-1.5)
            recent_factor = exp(-ln(2) / 720h * hours_since_last_access)
            effective_access = access_count * recent_factor
            resistance = 1 / (1 + effective_access * 0.1)
            new_salience = salience * exp(-decay_rate * hours * resistance)
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        coherence_factor = 0.5 + max(0.0, min(1.0, coherence))
        base_rate = math.log(2) / max(half_life_hours, 0.1)
        decay_rate = base_rate * coherence_factor

        # Recency half-life for access count weighting
        access_recency_rate = math.log(2) / self.ACCESS_RECENCY_HALF_LIFE_HOURS

        # Consolidation kinds get extra decay resistance — they're
        # distilled knowledge and should persist much longer.
        _CONSOLIDATION_KINDS = frozenset(("summary", "thread", "arc"))
        _CONSOLIDATION_RESISTANCE = 0.2  # 5x slower decay

        rows = self.conn.execute(
            "SELECT id, salience, access_count, last_accessed, kind FROM traces"
        ).fetchall()

        updates = []
        for row in rows:
            trace_id = row[0]
            salience = row[1]
            access_count = row[2] or 0
            last_accessed = row[3]
            kind = row[4] or ""

            if not last_accessed:
                continue

            try:
                last_dt = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
                # Strip timezone for comparison with utcnow
                last_dt = last_dt.replace(tzinfo=None)
            except (ValueError, AttributeError):
                continue

            hours_since = (now - last_dt).total_seconds() / 3600.0
            if hours_since <= 0:
                continue

            # ACT-R-inspired access recency: old accesses fade from
            # resistance calculation.  A trace accessed 10 times but
            # not touched in 60 days has effective_access ~ 2.5.
            recent_factor = math.exp(-access_recency_rate * hours_since)
            effective_access = access_count * recent_factor
            resistance = 1.0 / (1.0 + effective_access * 0.1)
            # Consolidation traces decay much slower
            if kind in _CONSOLIDATION_KINDS:
                resistance *= _CONSOLIDATION_RESISTANCE
            new_salience = salience * math.exp(-decay_rate * hours_since * resistance)
            new_salience = max(0.0, new_salience)

            if abs(new_salience - salience) > 1e-6:
                updates.append((new_salience, trace_id))

        if updates:
            self.conn.executemany(
                "UPDATE traces SET salience = ? WHERE id = ?", updates
            )
            self._commit()

    def prune(self, min_salience: float = 0.01):
        """Delete traces that have decayed below the minimum salience threshold.

        Consolidation traces (summary, thread, arc) are never pruned —
        they represent distilled knowledge whose children have already
        been marked ``consolidated_into`` and cannot be re-consolidated.
        """
        self.conn.execute(
            "DELETE FROM traces WHERE salience < ? "
            "AND kind NOT IN ('summary', 'thread', 'arc')",
            (min_salience,),
        )
        self._commit()

    # ── Sessions ──────────────────────────────────────────────

    def start_session(
        self,
        person: str,
        started: Optional[str] = None,
        project: str = "",
    ) -> str:
        """Create a new session. Returns session ID."""
        session_id = _generate_id()
        started = started or _now()
        self.conn.execute(
            """
            INSERT INTO sessions (id, person, started, ended, message_count,
                                  summary, metadata, project)
            VALUES (?, ?, ?, NULL, 0, NULL, NULL, ?)
            """,
            (session_id, person, started, project),
        )
        self._commit()
        return session_id

    def end_session(
        self,
        session_id: str,
        summary: str = "",
        ended: Optional[str] = None,
    ) -> None:
        """Mark a session as ended with optional summary."""
        ended = ended or _now()
        self.conn.execute(
            "UPDATE sessions SET ended = ?, summary = ? WHERE id = ?",
            (ended, summary or None, session_id),
        )
        self._commit()

    def get_active_session(self, person: str) -> Optional[Dict]:
        """Get the most recent session for a person that hasn't ended.

        Returns None if no active session exists.
        """
        row = self.conn.execute(
            """
            SELECT * FROM sessions
            WHERE person = ? AND ended IS NULL
            ORDER BY started DESC
            LIMIT 1
            """,
            (person,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_recent_sessions(
        self,
        person: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Get recent sessions, optionally filtered by person."""
        if person:
            rows = self.conn.execute(
                """
                SELECT * FROM sessions
                WHERE person = ?
                ORDER BY started DESC
                LIMIT ?
                """,
                (person, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM sessions
                ORDER BY started DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def increment_session_message_count(self, session_id: str) -> None:
        """Increment the message count for a session."""
        self.conn.execute(
            "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
            (session_id,),
        )
        self._commit()

    def detect_session_boundary(
        self,
        person: str,
        gap_hours: float = 2.0,
    ) -> bool:
        """Check if a new session should start based on time gap.

        Returns True if the last message from this person was more
        than ``gap_hours`` ago (or there are no messages), meaning
        we should start a new session.
        """
        row = self.conn.execute(
            """
            SELECT timestamp FROM messages
            WHERE person = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (person,),
        ).fetchone()
        if row is None:
            return True  # no messages → new session

        try:
            last_ts = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            gap = (now - last_ts).total_seconds() / 3600.0
            return gap >= gap_hours
        except (ValueError, TypeError, AttributeError):
            return True

    # ── Temporal retrieval ─────────────────────────────────────

    def get_messages_in_range(
        self,
        since: str,
        until: str,
        person: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Retrieve messages within a time range.

        Parameters
        ----------
        since : str
            ISO 8601 start timestamp.
        until : str
            ISO 8601 end timestamp.
        person : str, optional
            Filter by person.
        limit : int
            Maximum results.
        """
        if person:
            rows = self.conn.execute(
                """
                SELECT * FROM messages
                WHERE timestamp >= ? AND timestamp <= ? AND person = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (since, until, person, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM messages
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (since, until, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_traces_in_range(
        self,
        since: str,
        until: str,
        limit: int = 50,
    ) -> List[Dict]:
        """Retrieve traces created within a time range."""
        rows = self.conn.execute(
            """
            SELECT * FROM traces
            WHERE created >= ? AND created <= ?
            ORDER BY salience DESC
            LIMIT ?
            """,
            (since, until, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_sessions_in_range(
        self,
        since: str,
        until: str,
        person: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Retrieve sessions that overlap with a time range."""
        if person:
            rows = self.conn.execute(
                """
                SELECT * FROM sessions
                WHERE started <= ? AND (ended >= ? OR ended IS NULL) AND person = ?
                ORDER BY started DESC
                LIMIT ?
                """,
                (until, since, person, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM sessions
                WHERE started <= ? AND (ended >= ? OR ended IS NULL)
                ORDER BY started DESC
                LIMIT ?
                """,
                (until, since, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ── Trace-kind queries (temporal, utility, etc.) ─────────

    def get_traces_by_kind(
        self,
        kind: str,
        limit: int = 50,
        min_salience: float = 0.0,
        project: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve traces of a specific kind (temporal, utility, etc.).

        Metadata is automatically deserialized so callers can access
        temporal decay/revival data or utility Q-values directly.
        When ``project`` is provided, includes project-scoped + global traces.
        """
        if project is not None:
            rows = self.conn.execute(
                """
                SELECT * FROM traces
                WHERE kind = ? AND salience >= ?
                  AND (project = ? OR project = '')
                ORDER BY salience DESC
                LIMIT ?
                """,
                (kind, min_salience, project, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM traces
                WHERE kind = ? AND salience >= ?
                ORDER BY salience DESC
                LIMIT ?
                """,
                (kind, min_salience, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_traces_with_metadata(
        self,
        metadata_key: str,
        limit: int = 50,
    ) -> List[Dict]:
        """Retrieve traces where metadata contains a specific key.

        Uses SQLite ``json_extract`` for efficient filtering.
        Useful for finding all traces with temporal decay data,
        utility Q-values, belief scores, etc.
        """
        rows = self.conn.execute(
            """
            SELECT * FROM traces
            WHERE json_extract(metadata, ?) IS NOT NULL
            ORDER BY salience DESC
            LIMIT ?
            """,
            (f"$.{metadata_key}", limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_trace_metadata(
        self,
        trace_id: str,
        key: str,
        value: Any,
    ) -> bool:
        """Update a single key in a trace's metadata JSON.

        Creates the metadata JSON if it doesn't exist.
        Returns True if the trace was found and updated.
        """
        row = self.conn.execute(
            "SELECT metadata FROM traces WHERE id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            return False
        existing = {}
        if row[0]:
            try:
                existing = json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                existing = {}
        existing[key] = value
        self.conn.execute(
            "UPDATE traces SET metadata = ? WHERE id = ?",
            (json.dumps(existing), trace_id),
        )
        self._commit()
        return True

    # ── Relationship graph ────────────────────────────────────

    def add_relationship(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        source_trace_id: Optional[str] = None,
        **metadata,
    ) -> str:
        """Add a relationship edge to the knowledge graph.

        If an identical (subject, predicate, object) triple already
        exists and is currently valid (``valid_until IS NULL``), the
        existing ID is returned and confidence is updated to the
        maximum of old and new values.

        Parameters
        ----------
        subject : str
            Source entity name (e.g. "Thomas", "Python").
        predicate : str
            Relationship type (e.g. "created_by", "likes", "knows").
        object : str
            Target entity name (e.g. "Aidan", "async patterns").
        confidence : float
            Confidence in this relationship (0-1, default 1.0).
        source_trace_id : str, optional
            The episodic trace that sourced this relationship.
        **metadata :
            Additional key-value metadata.

        Returns
        -------
        str
            The relationship ID (new or existing).
        """
        # Check for existing active triple
        existing = self.conn.execute(
            """
            SELECT id, confidence FROM relationships
            WHERE subject = ? AND predicate = ? AND object = ?
              AND valid_until IS NULL
            LIMIT 1
            """,
            (subject, predicate, object),
        ).fetchone()

        if existing:
            # Update confidence if higher
            new_confidence = max(existing[1] or 0.0, confidence)
            self.conn.execute(
                "UPDATE relationships SET confidence = ? WHERE id = ?",
                (new_confidence, existing[0]),
            )
            self._commit()
            return existing[0]

        rel_id = _generate_id()
        self.conn.execute(
            """
            INSERT INTO relationships
                (id, subject, predicate, object, valid_from, valid_until,
                 confidence, source_trace_id, metadata)
            VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)
            """,
            (
                rel_id,
                subject,
                predicate,
                object,
                _now(),
                confidence,
                source_trace_id,
                json.dumps(metadata) if metadata else None,
            ),
        )
        self._commit()
        return rel_id

    def invalidate_relationship(self, rel_id: str) -> None:
        """Mark a relationship as no longer valid (temporal end)."""
        self.conn.execute(
            "UPDATE relationships SET valid_until = ? WHERE id = ?",
            (_now(), rel_id),
        )
        self._commit()

    def get_relationships(
        self,
        entity: str,
        direction: str = "both",
        predicate: Optional[str] = None,
        include_expired: bool = False,
    ) -> List[Dict]:
        """Get relationships for an entity (1-hop graph query).

        Parameters
        ----------
        entity : str
            Entity name to search for.
        direction : str
            ``"outgoing"`` (entity is subject), ``"incoming"`` (entity
            is object), or ``"both"`` (default).
        predicate : str, optional
            Filter by relationship type.
        include_expired : bool
            If True, include relationships with ``valid_until`` set.

        Returns
        -------
        list[dict]
            Relationship records sorted by confidence descending.
        """
        clauses: List[str] = []
        params: list = []

        if direction == "outgoing":
            clauses.append("subject = ?")
            params.append(entity)
        elif direction == "incoming":
            clauses.append("object = ?")
            params.append(entity)
        else:
            clauses.append("(subject = ? OR object = ?)")
            params.extend([entity, entity])

        if predicate:
            clauses.append("predicate = ?")
            params.append(predicate)

        if not include_expired:
            clauses.append("valid_until IS NULL")

        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM relationships{where} ORDER BY confidence DESC",
            params,
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_entity_graph(
        self,
        entity: str,
        hops: int = 1,
        include_expired: bool = False,
    ) -> List[Dict]:
        """Get multi-hop relationships from an entity.

        Parameters
        ----------
        entity : str
            Starting entity.
        hops : int
            Number of hops (1 = direct relationships, 2 = friends-of-friends).
        include_expired : bool
            Include expired relationships.

        Returns
        -------
        list[dict]
            All relationships within the hop radius.
        """
        seen_ids: set = set()
        entities_to_explore = {entity}
        all_relationships: List[Dict] = []

        for _ in range(hops):
            next_entities: set = set()
            for e in entities_to_explore:
                rels = self.get_relationships(
                    e, direction="both", include_expired=include_expired
                )
                for rel in rels:
                    if rel["id"] not in seen_ids:
                        seen_ids.add(rel["id"])
                        all_relationships.append(rel)
                        # Add the other end of the relationship
                        next_entities.add(rel["subject"])
                        next_entities.add(rel["object"])
            entities_to_explore = next_entities - {entity}

        return all_relationships

    def count_relationships(self, include_expired: bool = False) -> int:
        """Count relationships in the graph."""
        if include_expired:
            row = self.conn.execute("SELECT COUNT(*) FROM relationships").fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM relationships WHERE valid_until IS NULL"
            ).fetchone()
        return row[0] if row else 0

    # ── Internal ──────────────────────────────────────────────

    _VALID_TABLES = {"messages", "traces", "events"}

    def _validate_table(self, table: str):
        if table not in self._VALID_TABLES:
            raise ValueError(
                f"Invalid table '{table}'. Must be one of: {self._VALID_TABLES}"
            )

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
