"""
engram.system -- Top-level MemorySystem: the public API for Engram.

    from engram import MemorySystem

    memory = MemorySystem(data_dir="./data")
    context = memory.before(person="alice", message="hello")
    result  = memory.after(person="alice", their_message="hello",
                           response="Hi Alice!", trace_ids=context.trace_ids)

Everything is wired up here: stores, search, signal, pipelines.
The user only needs to touch MemorySystem.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from engram.core.config import Config
from engram.core.types import AfterResult, Context, LLMFunc, MemoryStats, Signal

# Sentinel: distinguishes "never tried" from "tried and failed"
_NOT_SET = object()
from engram.consciousness.boot import BootSequence
from engram.consciousness.identity import IdentityLoop
from engram.emotional import EmotionalSystem
from engram.episodic.store import EpisodicStore
from engram.introspection import IntrospectionLayer
from engram.journal import JournalStore
from engram.personality import PersonalitySystem
from engram.pipeline.after import after as _after_pipeline
from engram.pipeline.before import before as _before_pipeline
from engram.procedural.store import ProceduralStore
from engram.runtime.modes import ModeManager
from engram.safety import InfluenceLog, InjuryTracker
from engram.search.indexed import IndexedSearch
from engram.semantic.identity import IdentityResolver
from engram.semantic.store import SemanticStore
from engram.consolidation.compactor import ConversationCompactor
from engram.consolidation.consolidator import MemoryConsolidator
from engram.consolidation.pressure import MemoryPressure
from engram.signal.decay import DecayEngine
from engram.signal.measure import SignalTracker
from engram.signal.reinforcement import ReinforcementEngine
from engram.trust import TrustGate
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

        # -- Initialise stores ---------------------------------------------
        self.episodic = EpisodicStore(self.config.db_path)
        self.semantic = SemanticStore(
            semantic_dir=self.config.semantic_dir,
            soul_dir=self.config.soul_dir,
        )
        self.identity = IdentityResolver(self.config.identities_path)
        self.procedural = ProceduralStore(self.config.procedural_dir)

        # -- Journal, safety -----------------------------------------------
        self.journal = JournalStore(self.config.soul_dir / "journal")
        safety_dir = self.config.data_dir / "safety"
        self.influence = InfluenceLog(safety_dir)
        self.injury = InjuryTracker(safety_dir)

        # -- Search layer --------------------------------------------------
        self.indexed = IndexedSearch(self.config.db_path)

        self._semantic_search = None
        self._embedding_func = embedding_func
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

        # -- Trust gate -------------------------------------------------------
        self.trust_gate = TrustGate(
            semantic=self.semantic,
            core_person=self.config.core_person,
        )
        self.trust_gate.ensure_core_person()

        # -- Personality (Big Five) ----------------------------------------
        self.personality = PersonalitySystem(
            storage_dir=self.config.personality_dir,
        )

        # -- Emotional continuity (VAD) ------------------------------------
        self.emotional = EmotionalSystem(
            storage_dir=self.config.emotional_dir,
            valence_decay=self.config.emotional_valence_decay,
            arousal_decay=self.config.emotional_arousal_decay,
            dominance_decay=self.config.emotional_dominance_decay,
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

        # -- Introspection -------------------------------------------------
        self.introspection = IntrospectionLayer(
            storage_dir=self.config.introspection_dir,
            history_days=self.config.introspection_history_days,
        )

        # -- Consciousness: boot + identity loop ---------------------------
        self.boot_sequence = BootSequence(
            data_dir=self.config.data_dir,
            n_recent=self.config.boot_n_recent,
            n_key=self.config.boot_n_key,
            n_intro=self.config.boot_n_intro,
        )
        self.identity_loop = IdentityLoop(
            storage_dir=self.config.consciousness_dir,
        )

        # -- Runtime mode manager ------------------------------------------
        self.mode_manager = ModeManager(
            default_mode=self.config.runtime_default_mode,
        )

        # -- LLM function (lazy) -------------------------------------------
        self._llm_func: Any = _NOT_SET  # _NOT_SET | None | LLMFunc

        log.info(
            "MemorySystem initialized: data_dir=%s, provider=%s, model=%s",
            self.config.data_dir,
            self.config.llm_provider,
            self.config.llm_model,
        )

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
        """Lazily initialize the unified search (FTS + optional ChromaDB)."""
        if self._unified_search is None:
            from engram.search.unified import UnifiedSearch

            from engram.search.semantic import SemanticSearch

            sem = None
            try:
                sem = SemanticSearch(
                    embeddings_dir=self.config.embeddings_dir,
                    embedding_func=self._embedding_func,
                )
                self._semantic_search = sem
            except Exception as exc:
                log.warning("Failed to init semantic search: %s", exc)

            self._unified_search = UnifiedSearch(
                indexed=self.indexed,
                semantic=sem,
            )
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
            injury=self.injury,
            trust_gate=self.trust_gate,
            personality=self.personality,
            emotional=self.emotional,
            workspace=self.workspace,
            boot_sequence=self.boot_sequence,
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

        # Check if this person's trust tier allows memory persistence
        policy = self.trust_gate.policy_for(canonical, source=source)
        skip = not policy.memory_persistent

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
            skip_persistence=skip,
            emotional=self.emotional,
            introspection=self.introspection,
            identity_loop=self.identity_loop,
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
        Consciousness boot — load identity grounding context.

        Loads SOUL.md, recent high-salience emotional episodes, and
        anchoring beliefs into a single grounding context. Call at
        session start to establish identity coherence.

        Now also includes episodic boot priming (experiential memories),
        personality profile, and current emotional state.
        """
        soul_text = self.semantic.get_identity()

        # High-salience traces (the most important memories)
        top_traces = self.episodic.get_by_salience(limit=5)

        # Get preferences and boundaries as grounding
        prefs = self.semantic.get_preferences()
        boundaries = self.semantic.get_boundaries()

        # Anchoring beliefs from injury tracker
        anchors = self.injury.get_anchors()

        # Active injuries (need awareness)
        active_injuries = self.injury.get_status()

        # Recent journal entries (processed experience)
        recent_journal = self.journal.list_entries(limit=3)

        # Episodic boot priming — experiential memories
        boot_priming = ""
        try:
            boot_priming = self.boot_sequence.generate()
        except Exception as exc:
            log.debug("Boot priming generation failed: %s", exc)

        # Personality snapshot
        personality_report = {}
        try:
            personality_report = self.personality.report()
        except Exception as exc:
            log.debug("Personality report failed: %s", exc)

        # Emotional state
        emotional_state = {}
        try:
            emotional_state = self.emotional.current_state()
        except Exception as exc:
            log.debug("Emotional state failed: %s", exc)

        # Identity solidification
        identity_report = {}
        try:
            identity_report = self.identity_loop.solidification_report()
        except Exception as exc:
            log.debug("Identity report failed: %s", exc)

        # Switch to active mode on boot
        try:
            self.mode_manager.set_mode("active", reason="session boot")
        except Exception as exc:
            log.debug("Mode transition failed: %s", exc)

        return {
            "soul": soul_text[:3000] if soul_text else "",
            "top_memories": top_traces,
            "preferences_summary": prefs[:500] if prefs else "",
            "boundaries_summary": boundaries[:500] if boundaries else "",
            "anchoring_beliefs": anchors,
            "active_injuries": [
                {
                    "title": i.get("title", ""),
                    "status": i.get("status", ""),
                    "severity": i.get("severity", ""),
                }
                for i in active_injuries[:3]
            ],
            "recent_journal": recent_journal,
            "signal_health": self.signal_tracker.recent_health(),
            "signal_trend": self.signal_tracker.trend(),
            "boot_priming": boot_priming[:2000] if boot_priming else "",
            "personality": personality_report,
            "emotional_state": emotional_state,
            "identity_solidification": identity_report,
            "mode": self.mode_manager.status(),
        }

    def get_stats(self) -> MemoryStats:
        """Get memory system health statistics."""
        trace_count = self.episodic.count_traces()
        msg_count = self.episodic.count_messages()
        avg_sal = self.episodic.avg_salience("traces")
        skill_count = len(self.procedural.list_skills())
        rel_count = len(self.semantic.list_relationships())

        pressure = trace_count / max(1, self.config.max_traces)

        return MemoryStats(
            episodic_count=trace_count,
            semantic_facts=rel_count,
            procedural_skills=skill_count,
            total_messages=msg_count,
            avg_salience=avg_sal,
            memory_pressure=pressure,
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
