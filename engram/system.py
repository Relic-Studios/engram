"""
engram.system -- Top-level MemorySystem: the public API for Engram.

    from engram import MemorySystem

    memory = MemorySystem(data_dir="./data")
    context = memory.before(person="alice", message="hello")
    result  = memory.after(person="alice", their_message="hello",
                           response="Hi Alice!", trace_ids=context.trace_ids)

Everything is wired up here: stores, search, signal, pipelines.
The user only needs to touch MemorySystem.

v2 code-first pivot: consciousness modules (personality, emotional,
introspection, identity, trust, safety, boot) removed. Retrieval
pipeline preserved intact.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from engram.core.config import Config
from engram.core.types import AfterResult, Context, LLMFunc, MemoryStats, Signal
from engram.working.allocator import reorder_u

# Sentinel: distinguishes "never tried" from "tried and failed"
_NOT_SET = object()

from engram.episodic.store import EpisodicStore
from engram.journal import JournalStore
from engram.pipeline.after import after as _after_pipeline
from engram.pipeline.before import before as _before_pipeline
from engram.procedural.store import ProceduralStore
from engram.search.indexed import IndexedSearch
from engram.semantic.identity import IdentityResolver
from engram.semantic.store import SemanticStore
from engram.consolidation.compactor import ConversationCompactor
from engram.consolidation.consolidator import MemoryConsolidator
from engram.consolidation.pressure import MemoryPressure
from engram.signal.decay import DecayEngine
from engram.signal.measure import SignalTracker
from engram.signal.reinforcement import ReinforcementEngine
from engram.working.context import ContextBuilder
from engram.workspace import CognitiveWorkspace

log = logging.getLogger("engram.system")


class MemorySystem:
    """Top-level API: wire everything together and expose before/after.

    Parameters
    ----------
    config:
        Full ``Config`` object.  If not given, ``data_dir`` and
        ``**kwargs`` are forwarded to ``Config``.
    data_dir:
        Shortcut -- if you just want to point at a directory and go.
    embedding_func:
        Optional custom embedding function for semantic search.
        Signature: ``(text: str) -> list[float]``.
    **kwargs:
        Extra keyword args forwarded to ``Config()``.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        data_dir: Optional[str | Path] = None,
        embedding_func: Optional[Callable[..., list]] = None,
        **kwargs: Any,
    ) -> None:
        # -- Resolve config ------------------------------------------------
        if config is not None:
            self.config = config
        elif data_dir is not None:
            self.config = Config.from_data_dir(data_dir, **kwargs)
        else:
            self.config = Config(**kwargs)

        self.config.ensure_directories()

        # -- Structured logging (OB-3) ------------------------------------
        if self.config.structured_logging:
            from engram.core.logging import configure_logging

            configure_logging(structured=True)

        # -- Initialise stores ---------------------------------------------
        self.episodic = EpisodicStore(self.config.db_path)
        self.semantic = SemanticStore(
            semantic_dir=self.config.semantic_dir,
            soul_dir=self.config.soul_dir,
        )
        self.identity = IdentityResolver(self.config.identities_path)
        self.procedural = ProceduralStore(self.config.procedural_dir)

        # -- Journal -------------------------------------------------------
        self.journal = JournalStore(self.config.soul_dir / "journal")

        # -- Search layer --------------------------------------------------
        self.indexed = IndexedSearch(self.config.db_path)

        self._semantic_search = None
        # Use explicitly provided embedding_func, or fall back to
        # config.get_embedding_func() which builds one from the
        # configured embedding_model (if set).
        if embedding_func is not None:
            self._embedding_func = embedding_func
        else:
            try:
                self._embedding_func = self.config.get_embedding_func()
            except Exception as exc:
                log.warning("Failed to build embedding function: %s", exc)
                self._embedding_func = None

        # Code-specific embedding function (dual-embedding space).
        # Built lazily from config.  If unavailable, code content
        # is only indexed with NL embeddings (graceful degradation).
        self._code_embedding_func = None
        try:
            self._code_embedding_func = self.config.get_code_embedding_func()
        except Exception as exc:
            log.info("Code embeddings not available: %s", exc)

        # Defer semantic search init (ChromaDB) until first use
        # since it's optional and heavyweight.
        self._unified_search = None

        # -- Signal & learning subsystems ----------------------------------
        self.signal_tracker = SignalTracker(window_size=50)
        self.reinforcement = ReinforcementEngine(
            reinforce_delta=self.config.reinforce_delta,
            weaken_delta=self.config.weaken_delta,
            reinforce_threshold=self.config.reinforce_threshold,
            weaken_threshold=self.config.weaken_threshold,
        )
        self.decay_engine = DecayEngine(
            half_life_hours=self.config.decay_half_life_hours,
        )

        # -- Context builder -----------------------------------------------
        self.context_builder = ContextBuilder(
            token_budget=self.config.token_budget,
            config=self.config,
        )

        # -- Consolidation (MemGPT-inspired) --------------------------------
        self.memory_pressure = MemoryPressure(self.config)
        self.compactor = ConversationCompactor(
            keep_recent=self.config.compaction_keep_recent,
            segment_size=self.config.compaction_segment_size,
            min_messages_to_compact=self.config.compaction_min_messages,
        )
        self.consolidator = MemoryConsolidator(
            min_episodes_per_thread=self.config.consolidation_min_episodes,
            thread_time_window_hours=self.config.consolidation_time_window_hours,
            min_threads_per_arc=self.config.consolidation_min_threads,
            max_episodes_per_run=self.config.consolidation_max_episodes_per_run,
        )

        # -- Cognitive workspace (7±2) -------------------------------------
        self.workspace = CognitiveWorkspace(
            capacity=self.config.workspace_capacity,
            decay_rate=self.config.workspace_decay_rate,
            rehearsal_boost=self.config.workspace_rehearsal_boost,
            expiry_threshold=self.config.workspace_expiry_threshold,
            storage_path=self.config.workspace_path,
            on_evict=self._workspace_evict_callback,
        )

        # -- Incremental vector indexing (FR-2) ----------------------------
        # When a new trace is logged to episodic memory, push it into
        # ChromaDB so semantic search stays in sync with ground truth.
        self.episodic._on_trace_logged = self._index_trace_callback

        # -- LLM function (lazy) -------------------------------------------
        self._llm_func: Any = _NOT_SET  # _NOT_SET | None | LLMFunc

        log.info(
            "MemorySystem initialized: data_dir=%s, provider=%s, model=%s",
            self.config.data_dir,
            self.config.llm_provider,
            self.config.llm_model,
        )

    def _index_trace_callback(
        self, trace_id: str, content: str, metadata: Dict
    ) -> None:
        """Push a newly logged trace into ChromaDB for vector search.

        Called by ``EpisodicStore.log_trace()`` via the
        ``_on_trace_logged`` callback.  Accessing ``self.unified_search``
        lazily initialises ChromaDB if it hasn't been created yet.

        For code-related traces, dual-indexes in both the episodic (NL)
        collection and the code collection (code embeddings).
        """
        try:
            sem = self._semantic_search
            if sem is None:
                # Force lazy init so ChromaDB is available
                _ = self.unified_search
                sem = self._semantic_search
            if sem is not None:
                # Always index in episodic (NL embeddings)
                sem.index_trace(trace_id, content, metadata)

                # Dual-index code content in code collection
                if sem.has_code_embeddings:
                    from engram.search.code_embeddings import is_code_content

                    kind = metadata.get("kind", "")
                    if is_code_content(trace_kind=kind, content=content):
                        code_meta = dict(metadata)
                        code_meta["trace_id"] = trace_id
                        sem.index_code(f"code_{trace_id}", content, code_meta)
        except Exception as exc:
            log.debug("Incremental vector indexing failed for %s: %s", trace_id, exc)

    def _workspace_evict_callback(self, slot_data: Dict) -> None:
        """Route evicted workspace items to episodic store as traces."""
        try:
            self.episodic.log_trace(
                content=slot_data.get("content", ""),
                kind="workspace_eviction",
                tags=["workspace", slot_data.get("source", "unknown")],
                salience=max(0.1, slot_data.get("priority_when_removed", 0.2)),
                removal_reason=slot_data.get("removal_reason", ""),
                access_count=slot_data.get("access_count", 0),
                time_in_workspace=slot_data.get("time_in_workspace", 0),
            )
        except Exception as exc:
            log.debug("Workspace eviction logging failed: %s", exc)

    # ------------------------------------------------------------------
    # Lazy initializers
    # ------------------------------------------------------------------

    @property
    def llm_func(self) -> Optional[LLMFunc]:
        """Lazily build the LLM callable from config.

        Uses a sentinel to distinguish "never tried" from "tried and
        failed".  If initialisation fails once, ``None`` is cached so
        we don't retry (and spam warnings) on every access.
        """
        if self._llm_func is _NOT_SET:
            try:
                self._llm_func = self.config.get_llm_func()
            except Exception as exc:
                log.warning("Could not build LLM func: %s", exc)
                self._llm_func = None  # cache failure — don't retry
        return self._llm_func

    @property
    def unified_search(self):
        """Lazily initialize the unified search (FTS + optional ChromaDB + reranker)."""
        if self._unified_search is None:
            from engram.search.unified import UnifiedSearch

            from engram.search.semantic import SemanticSearch

            sem = None
            try:
                sem = SemanticSearch(
                    embeddings_dir=self.config.embeddings_dir,
                    embedding_func=self._embedding_func,
                    code_embedding_func=self._code_embedding_func,
                )
                self._semantic_search = sem
                if sem.has_code_embeddings:
                    log.info("Dual-embedding space active (NL + code)")
            except Exception as exc:
                log.warning("Failed to init semantic search: %s", exc)

            # Cross-encoder reranker (optional, best-effort)
            reranker = None
            if self.config.reranker_enabled:
                from engram.search.reranker import Reranker

                reranker = Reranker(
                    model_name=self.config.reranker_model,
                    device=self.config.reranker_device or None,
                )

            self._unified_search = UnifiedSearch(
                indexed=self.indexed,
                semantic=sem,
                reranker=reranker,
            )

            # Wire semantic search into the consolidator for
            # topic-coherent HDBSCAN clustering.
            if sem is not None and hasattr(self, "consolidator"):
                self.consolidator._semantic_search = sem

        return self._unified_search

    # ------------------------------------------------------------------
    # Pipeline: before (context injection)
    # ------------------------------------------------------------------

    def before(
        self,
        person: str,
        message: str,
        source: str = "direct",
        token_budget: Optional[int] = None,
    ) -> Context:
        """Run the before-pipeline: resolve identity, load context.

        Parameters
        ----------
        person:
            Raw identifier (alias, Discord handle, etc.).
        message:
            The incoming message.
        source:
            Where the message came from.
        token_budget:
            Override the default token budget for this call.

        Returns
        -------
        Context
            Assembled context ready for system-prompt injection.
        """
        builder = self.context_builder
        if token_budget is not None:
            # Create a per-call builder to avoid mutating shared state
            from engram.working.context import ContextBuilder

            builder = ContextBuilder(token_budget=token_budget, config=self.config)

        # Accessing unified_search lazily initialises _semantic_search,
        # so reference it first to ensure ChromaDB is available.
        _ = self.unified_search

        return _before_pipeline(
            person_raw=person,
            message=message,
            source=source,
            config=self.config,
            identity=self.identity,
            semantic=self.semantic,
            episodic=self.episodic,
            procedural=self.procedural,
            context_builder=builder,
            search=self.unified_search,
            signal_tracker=self.signal_tracker,
            journal=self.journal,
            workspace=self.workspace,
            semantic_search=self._semantic_search,
        )

    # ------------------------------------------------------------------
    # Pipeline: after (logging, learning)
    # ------------------------------------------------------------------

    def after(
        self,
        person: str,
        their_message: str,
        response: str,
        source: str = "direct",
        trace_ids: Optional[List[str]] = None,
    ) -> AfterResult:
        """Run the after-pipeline: measure signal, log, learn.

        Parameters
        ----------
        person:
            Canonical name (or raw alias -- will be resolved).
        their_message:
            What they said.
        response:
            What we replied.
        source:
            Message source identifier.
        trace_ids:
            Trace IDs from the before-pipeline context.

        Returns
        -------
        AfterResult
            Signal, salience, updates applied, logged IDs.
        """
        # Resolve person in case a raw alias was passed
        canonical = self.identity.resolve(person)

        return _after_pipeline(
            person=canonical,
            their_message=their_message,
            response=response,
            source=source,
            trace_ids=trace_ids,
            config=self.config,
            episodic=self.episodic,
            semantic=self.semantic,
            procedural=self.procedural,
            reinforcement=self.reinforcement,
            decay_engine=self.decay_engine,
            signal_tracker=self.signal_tracker,
            llm_func=self.llm_func,
            memory_pressure=self.memory_pressure,
            compactor=self.compactor,
            consolidator=self.consolidator,
            skip_persistence=False,
            workspace=self.workspace,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        person: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Search across all memory types."""
        canonical = self.identity.resolve(person) if person else None
        return self.unified_search.search(
            query=query,
            person=canonical,
            limit=limit,
        )

    def get_relationship(self, person: str) -> Optional[str]:
        """Get a person's relationship file content."""
        canonical = self.identity.resolve(person)
        return self.semantic.get_relationship(canonical)

    def get_identity(self) -> str:
        """Get the SOUL.md identity document."""
        return self.semantic.get_identity()

    def get_signal(self) -> Dict:
        """Get current signal tracker state."""
        return self.signal_tracker.to_dict()

    def boot(self) -> Dict:
        """
        Code-first boot — load project context and session priming data.

        Loads SOUL.md (coding philosophy), active ADRs (architecture
        decision records), recent high-salience traces, coding style
        preferences, active workspace items, recent sessions, and
        code quality signal health.

        This is the "contextual priming" step from the buildplan:
        ensures the agent starts each session with full awareness of
        the project's architecture, active decisions, and recent work.
        """
        soul_text = self.semantic.get_identity()

        # --- ADRs: highest-priority project context ---
        # Architecture decisions are assigned high initial salience
        # and should always be present during relevant tasks.
        adrs = self.episodic.get_traces_by_kind(
            "architecture_decision", limit=5, min_salience=0.3
        )
        # Primacy-recency: most important ADRs at start and end
        adrs = reorder_u(adrs, key_field="salience")

        # --- High-salience traces (code patterns, debug sessions, etc) ---
        top_traces = self.episodic.get_by_salience(limit=10)
        # Primacy-recency: highest-salience traces at start and end
        top_traces = reorder_u(top_traces, key_field="salience")

        # --- Coding style preferences ---
        prefs = self.semantic.get_preferences()

        # --- Recent journal entries (processed reflections) ---
        recent_journal = self.journal.list_entries(limit=3)

        # --- Recent sessions (what was worked on recently) ---
        recent_sessions = self.episodic.get_recent_sessions(
            limit=self.config.boot_n_sessions
        )

        # --- Active workspace items (current mental context) ---
        workspace_items = []
        try:
            workspace_items = self.workspace.items(n=5)
        except Exception:
            pass

        # --- Code quality signal health ---
        signal_health = self.signal_tracker.recent_health()
        signal_trend = self.signal_tracker.trend()

        return {
            "soul": soul_text[:3000] if soul_text else "",
            "architecture_decisions": adrs,
            "top_memories": top_traces,
            "preferences_summary": prefs[:500] if prefs else "",
            "recent_journal": recent_journal,
            "recent_sessions": recent_sessions,
            "workspace_items": workspace_items,
            "signal_health": signal_health,
            "signal_trend": signal_trend,
        }

    def get_stats(self) -> MemoryStats:
        """Get memory system health statistics."""
        trace_count = self.episodic.count_traces()
        msg_count = self.episodic.count_messages()
        avg_sal = self.episodic.avg_salience("traces")
        skill_count = len(self.procedural.list_skills())
        rel_count = len(self.semantic.list_relationships())

        pressure = trace_count / max(1, self.config.max_traces)

        # Check code embedding status
        code_emb_active = (
            self._semantic_search is not None
            and self._semantic_search.has_code_embeddings
        )

        return MemoryStats(
            episodic_count=trace_count,
            semantic_facts=rel_count,
            procedural_skills=skill_count,
            total_messages=msg_count,
            avg_salience=avg_sal,
            memory_pressure=pressure,
            code_embeddings_active=code_emb_active,
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def reindex(self) -> Dict[str, int]:
        """Full reindex of all memory layers into semantic search."""
        if self._semantic_search is None:
            # Force init
            _ = self.unified_search
        if self._semantic_search is not None:
            return self._semantic_search.reindex_all(
                episodic_store=self.episodic,
                semantic_store=self.semantic,
                procedural_store=self.procedural,
            )
        return {}

    def decay_pass(self) -> None:
        """Manually trigger an adaptive decay pass."""
        coherence = self.signal_tracker.recent_health()
        self.decay_engine.update_coherence(coherence)
        self.episodic.decay_pass(
            half_life_hours=self.config.decay_half_life_hours,
            coherence=coherence,
        )
        self.episodic.prune(min_salience=self.decay_engine.min_salience)

    def consolidate(self) -> Dict[str, List[str]]:
        """Run hierarchical memory consolidation.

        Groups episodes into threads, threads into arcs.
        Returns ``{"threads": [...ids], "arcs": [...ids]}``.
        """
        return self.consolidator.consolidate(
            episodic=self.episodic,
            llm_func=self.llm_func,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all database connections and release resources."""
        for name, resource in [
            ("episodic", self.episodic),
            ("indexed", self.indexed),
            ("semantic_search", self._semantic_search),
        ]:
            try:
                if resource is not None and hasattr(resource, "close"):
                    resource.close()
            except Exception as exc:
                log.debug("Error closing %s: %s", name, exc)

    def __enter__(self) -> "MemorySystem":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
