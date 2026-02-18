"""
engram.pipeline.before -- Pre-LLM context injection pipeline.

Called automatically before every LLM call.  Performs:
  1. Identity resolution (alias -> canonical name)
  2. Load identity (SOUL.md)
  3. Load relationship context for the person
  4. Assemble grounding context (preferences, boundaries,
     contradictions, recent journal)
  5. Fetch recent conversation history
  6. Fetch high-salience episodic traces (greedy knapsack)
  7. Match procedural skills
  8. Assemble everything into a token-budgeted Context object

The caller injects ``context.text`` into the system prompt.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional

from engram.core.types import Context

if TYPE_CHECKING:
    from engram.core.config import Config
    from engram.episodic.store import EpisodicStore
    from engram.journal import JournalStore
    from engram.procedural.store import ProceduralStore
    from engram.search.semantic import SemanticSearch
    from engram.search.unified import UnifiedSearch
    from engram.semantic.identity import IdentityResolver
    from engram.semantic.store import SemanticStore
    from engram.signal.measure import SignalTracker
    from engram.working.context import ContextBuilder
    from engram.workspace import CognitiveWorkspace

log = logging.getLogger("engram.pipeline.before")


def before(
    *,
    person_raw: str,
    message: str,
    source: str = "direct",
    config: "Config",
    identity: "IdentityResolver",
    semantic: "SemanticStore",
    episodic: "EpisodicStore",
    procedural: "ProceduralStore",
    context_builder: "ContextBuilder",
    search: Optional["UnifiedSearch"] = None,
    signal_tracker: Optional["SignalTracker"] = None,
    journal: Optional["JournalStore"] = None,
    workspace: Optional["CognitiveWorkspace"] = None,
    semantic_search: Optional["SemanticSearch"] = None,
) -> Context:
    """Run the full before-pipeline and return an assembled Context.

    Parameters
    ----------
    person_raw:
        Raw identifier for the conversation partner (Discord handle,
        nickname, etc.).  Will be resolved to a canonical name.
    message:
        The incoming message from the person.
    source:
        Where the message came from (``"discord"``, ``"opencode"``, etc.).
    config:
        Engram configuration.
    identity:
        Identity resolver for alias -> canonical mapping.
    semantic:
        Semantic memory store (SOUL.md, relationships, preferences).
    episodic:
        Episodic memory store (SQLite).
    procedural:
        Procedural memory store (skill files).
    context_builder:
        Working memory context builder.
    search:
        Optional unified search for RAG-style recall.
    signal_tracker:
        Optional signal tracker -- used to check if a correction
        prompt is needed (recent health drop).
    journal:
        Optional journal store for recent reflections.
    workspace:
        Optional cognitive workspace.  When provided, active workspace
        items are added to the message (high-priority mental context).
    semantic_search:
        Optional semantic search.  When provided, trace embeddings are
        fetched for MMR diversity in the knapsack allocator.

    Returns
    -------
    Context
        Assembled context with ``text``, ``trace_ids``, ``person``,
        token usage metadata.
    """
    timings: Dict[str, float] = {}
    _t0_total = time.perf_counter()

    # -- 1. Resolve identity -----------------------------------------------
    _t0 = time.perf_counter()
    person = identity.resolve(person_raw)
    timings["identity_resolve"] = time.perf_counter() - _t0
    log.debug("Resolved %r -> %r", person_raw, person)

    # -- 2. Load identity --------------------------------------------------
    identity_text = semantic.get_identity()

    # -- 3. Load relationship context --------------------------------------
    relationship_text = semantic.get_relationship(person) or ""

    # -- 4. Assemble grounding context -------------------------------------
    grounding_context = _build_grounding_context(
        person=person,
        semantic=semantic,
        journal=journal,
    )

    # -- 5. Recent conversation history ------------------------------------
    recent_messages = episodic.get_recent_messages(person=person, limit=20)

    # -- 6. High-salience episodic traces ----------------------------------
    salient_traces = episodic.get_by_salience(person=person, limit=30)

    # Supplement with search-based recall if available and the message
    # contains enough substance to search on.
    if search and len(message.split()) >= 3:
        try:
            _t0 = time.perf_counter()
            search_results = search.search(query=message, person=person, limit=10)
            # Merge search results into salient_traces, deduplicating by id
            existing_ids = {t.get("id") for t in salient_traces if t.get("id")}
            for result in search_results:
                rid = result.get("id") or result.get("trace_id") or result.get("doc_id")
                if rid and rid not in existing_ids:
                    # Normalize search result to trace-like dict
                    salient_traces.append(
                        {
                            "id": rid,
                            "content": result.get("content", ""),
                            "salience": _rrf_to_salience(
                                result.get("combined_score", 0.0)
                            ),
                            "kind": result.get("source", "episode"),
                        }
                    )
                    existing_ids.add(rid)
            timings["search_recall"] = time.perf_counter() - _t0
        except Exception as exc:
            log.debug("Search-based recall failed: %s", exc)

    # -- 7. Procedural skill matching --------------------------------------
    relevant_skills = procedural.match_context(message)

    # -- 8. Correction prompt (if recent signal health dipped) -------------
    correction_prompt = None
    if signal_tracker is not None:
        recent_health = signal_tracker.recent_health()
        if recent_health < config.signal_health_threshold:
            weakest = ""
            if signal_tracker.signals:
                weakest = signal_tracker.signals[-1].weakest_facet
            correction_prompt = _build_correction(recent_health, weakest)

    # -- 8b. Workspace items as high-priority mental context ---------------
    #    Active workspace items are prepended to the message context so
    #    the LLM is aware of what's currently "in mind".
    workspace_text = ""
    if workspace is not None:
        try:
            items = workspace.items(n=5)
            if items:
                workspace_text = "Currently in working memory:\n" + "\n".join(
                    f"- {item}" for item in items
                )
        except Exception as exc:
            log.debug("Workspace items failed: %s", exc)

    if workspace_text:
        grounding_context = (
            grounding_context + "\n\n" + workspace_text
            if grounding_context
            else workspace_text
        )

    # -- 8e. Fetch trace embeddings for MMR diversity -------------------------
    #    Pre-computed embeddings from ChromaDB allow the knapsack allocator
    #    to penalise near-duplicate traces via Maximal Marginal Relevance.
    trace_embeddings = None
    if semantic_search is not None and salient_traces:
        try:
            _t0 = time.perf_counter()
            # ChromaDB stores traces with "trace_" prefix on IDs
            trace_ids_for_emb = [
                f"trace_{t['id']}" for t in salient_traces if t.get("id")
            ]
            if trace_ids_for_emb:
                raw_embs = semantic_search.get_embeddings(
                    trace_ids_for_emb, collection="episodic"
                )
                # Map back to bare trace IDs (strip "trace_" prefix)
                trace_embeddings = {
                    k.removeprefix("trace_"): v for k, v in raw_embs.items()
                }
            timings["fetch_embeddings"] = time.perf_counter() - _t0
        except Exception as exc:
            log.debug("Failed to fetch trace embeddings for MMR: %s", exc)

    # -- 9. Assemble context -----------------------------------------------
    _t0 = time.perf_counter()
    ctx = context_builder.build(
        person=person,
        message=message,
        identity_text=identity_text,
        relationship_text=relationship_text,
        grounding_context=grounding_context,
        recent_messages=recent_messages,
        salient_traces=salient_traces,
        relevant_skills=relevant_skills,
        correction_prompt=correction_prompt,
        trace_embeddings=trace_embeddings,
    )

    timings["context_build"] = time.perf_counter() - _t0

    # Mark loaded traces as accessed (for decay resistance)
    for tid in ctx.trace_ids:
        try:
            episodic.update_access("traces", tid)
        except Exception as exc:
            log.debug("Failed to update access for trace %s: %s", tid, exc)

    timings["total"] = time.perf_counter() - _t0_total
    ctx.timings = timings

    log.info(
        "Before pipeline: person=%s, tokens=%d/%d, memories=%d, traces=%d, "
        "total_ms=%.1f",
        person,
        ctx.tokens_used,
        ctx.token_budget,
        ctx.memories_loaded,
        len(ctx.trace_ids),
        timings["total"] * 1000,
    )

    return ctx


# ---------------------------------------------------------------------------
# RRF score conversion
# ---------------------------------------------------------------------------


def _rrf_to_salience(rrf_score: float, k: int = 60) -> float:
    """Convert an RRF combined_score to a [0, 1] salience value.

    The maximum RRF score is 2/(k+1) when a document is ranked #1 in
    both FTS and semantic search.  We normalise to [0, 1] by dividing
    by this theoretical maximum.
    """
    max_rrf = 2.0 / (k + 1)
    if max_rrf <= 0:
        return 0.5
    return min(1.0, rrf_score / max_rrf)


# ---------------------------------------------------------------------------
# Grounding context builder
# ---------------------------------------------------------------------------


def _build_grounding_context(
    *,
    person: str,
    semantic: "SemanticStore",
    journal: Optional["JournalStore"] = None,
) -> str:
    """Assemble grounding context from semantic sources.

    This is always included in the context regardless of whether a
    relationship file exists.  It represents the agent's knowledge base:

    - Preferences (likes/dislikes/uncertainties)
    - Boundaries (behavioral limits)
    - Contradictions (tensions being held)
    - Recent journal (processed reflections)
    """
    parts: list[str] = []

    # Preferences
    try:
        prefs = semantic.get_preferences()
        if prefs:
            parts.append(f"My preferences:\n{prefs}")
    except Exception as exc:
        log.debug("Failed to load preferences: %s", exc)

    # Boundaries
    try:
        bounds = semantic.get_boundaries()
        if bounds:
            parts.append(f"My boundaries:\n{bounds}")
    except Exception as exc:
        log.debug("Failed to load boundaries: %s", exc)

    # Contradictions
    try:
        contradictions = semantic.get_contradictions()
        if contradictions:
            parts.append(f"Tensions I'm sitting with:\n{contradictions}")
    except Exception as exc:
        log.debug("Failed to load contradictions: %s", exc)

    # Recent journal entries
    if journal is not None:
        try:
            entries = journal.list_entries(limit=3)
            if entries:
                journal_lines = []
                for entry in entries:
                    topic = entry.get("topic", "?")
                    date = entry.get("date", "")
                    journal_lines.append(f"- {topic} ({date})")
                parts.append("Recent reflections:\n" + "\n".join(journal_lines))
        except Exception as exc:
            log.debug("Failed to load journal entries: %s", exc)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Correction prompt builder
# ---------------------------------------------------------------------------


def _build_correction(health: float, weakest_facet: str) -> str:
    """Build a correction nudge based on code quality signal health."""
    advice = {
        "correctness": (
            "Recent outputs have had correctness issues -- "
            "check for logic errors, undefined references, and "
            "incorrect API usage before responding."
        ),
        "consistency": (
            "Recent outputs have been inconsistent with project patterns -- "
            "follow established naming conventions, import styles, and "
            "architectural patterns."
        ),
        "completeness": (
            "Recent outputs have been incomplete -- "
            "include error handling, edge cases, type annotations, "
            "and all necessary imports."
        ),
        "robustness": (
            "Recent outputs have lacked robustness -- "
            "add input validation, null checks, proper error handling, "
            "and consider failure modes."
        ),
    }

    hint = advice.get(weakest_facet, "Review code quality before responding.")
    return (
        f"[Code quality signal: {health:.2f} -- {weakest_facet} is low]\n"
        f"{hint}\n"
        f"Focus on improving this aspect in your next response."
    )
