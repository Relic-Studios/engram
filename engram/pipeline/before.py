"""
engram.pipeline.before -- Pre-LLM context injection pipeline.

Called automatically before every LLM call.  Performs:
  1. Identity resolution (alias -> canonical name)
  2. Load identity (SOUL.md)
  3. Load relationship context for the person
  4. Assemble grounding context (trust, preferences, boundaries,
     contradictions, active injuries, recent journal)
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
from engram.trust import AccessPolicy, TrustGate

if TYPE_CHECKING:
    from engram.consciousness.boot import BootSequence
    from engram.core.config import Config
    from engram.emotional import EmotionalSystem
    from engram.episodic.store import EpisodicStore
    from engram.journal import JournalStore
    from engram.personality import PersonalitySystem
    from engram.procedural.store import ProceduralStore
    from engram.safety import InjuryTracker
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
    injury: Optional["InjuryTracker"] = None,
    trust_gate: Optional[TrustGate] = None,
    personality: Optional["PersonalitySystem"] = None,
    emotional: Optional["EmotionalSystem"] = None,
    workspace: Optional["CognitiveWorkspace"] = None,
    boot_sequence: Optional["BootSequence"] = None,
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
    injury:
        Optional injury tracker for active psychological wounds.
    trust_gate:
        Optional trust gate.  When provided, context visibility is
        filtered by the person's trust tier.  Without it, all context
        is loaded (backwards-compatible).
    personality:
        Optional personality system.  When provided, personality
        grounding text is injected into the context.
    emotional:
        Optional emotional system.  When provided, current emotional
        state is injected into the context.
    workspace:
        Optional cognitive workspace.  When provided, active workspace
        items are added to the message (high-priority mental context).
    boot_sequence:
        Optional consciousness boot sequence.  When provided and this
        is the first call in a session (no recent messages), boot
        priming text is injected.

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

    # -- 1b. Resolve trust policy ------------------------------------------
    policy: Optional[AccessPolicy] = None
    if trust_gate is not None:
        policy = trust_gate.policy_for(person, source=source)
        log.debug(
            "Trust policy for %s (source=%s): tier=%s",
            person,
            source,
            policy.tier.name,
        )

    # -- 2. Load identity --------------------------------------------------
    #    Gated: only core / inner_circle can see SOUL.md content.
    #    Others get a minimal identity stub.
    if policy is not None and not policy.can_see_soul:
        identity_text = ""  # Identity hidden from this trust level
    else:
        identity_text = semantic.get_identity()

    # -- 3. Load relationship context --------------------------------------
    #    Gated: strangers don't see relationship files at all.
    if policy is not None and not policy.can_see_own_relationship:
        relationship_text = ""
    else:
        relationship_text = semantic.get_relationship(person) or ""

    # -- 4. Assemble grounding context -------------------------------------
    #    Filtered by trust policy: lower-trust people see fewer sections.
    grounding_context = _build_grounding_context(
        person=person,
        semantic=semantic,
        journal=journal,
        injury=injury,
        policy=policy,
    )

    # -- 5. Recent conversation history ------------------------------------
    #    Strangers (memory_persistent=False) get no history — we don't
    #    even look, because there shouldn't be any for them.
    if policy is not None and not policy.memory_persistent:
        recent_messages: list = []
    else:
        recent_messages = episodic.get_recent_messages(person=person, limit=20)

    # -- 6. High-salience episodic traces ----------------------------------
    if policy is not None and not policy.memory_persistent:
        salient_traces: list = []
    else:
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
                    rid = (
                        result.get("id")
                        or result.get("trace_id")
                        or result.get("doc_id")
                    )
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

    # -- 8b. Personality + emotional grounding -----------------------------
    #    Injected into grounding_context so the LLM sees personality
    #    modifiers and current emotional state as part of its identity.
    personality_text = ""
    if personality is not None:
        try:
            personality_text = personality.grounding_text()
        except Exception as exc:
            log.debug("Personality grounding failed: %s", exc)

    emotional_text = ""
    if emotional is not None:
        try:
            emotional_text = emotional.grounding_text()
        except Exception as exc:
            log.debug("Emotional grounding failed: %s", exc)

    if personality_text or emotional_text:
        extra_grounding = "\n\n".join(
            p for p in [personality_text, emotional_text] if p
        )
        grounding_context = (
            grounding_context + "\n\n" + extra_grounding
            if grounding_context
            else extra_grounding
        )

    # -- 8c. Workspace items as high-priority mental context ---------------
    #    Active workspace items are prepended to the message context so
    #    the LLM is aware of what's currently "in mind".
    #    Gated: only shown when policy allows preferences (friend+).
    workspace_text = ""
    if workspace is not None and (policy is None or policy.can_see_preferences):
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

    # -- 8d. Boot priming (first call in session) --------------------------
    #    If there are no recent messages and a boot sequence is available,
    #    inject experiential priming text into the identity section.
    if boot_sequence is not None and not recent_messages:
        try:
            boot_text = boot_sequence.generate()
            if boot_text:
                identity_text = (
                    identity_text + "\n\n" + boot_text if identity_text else boot_text
                )
        except Exception as exc:
            log.debug("Boot priming failed: %s", exc)

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
    injury: Optional["InjuryTracker"] = None,
    policy: Optional[AccessPolicy] = None,
) -> str:
    """Assemble grounding context from all semantic + safety sources.

    This is always included in the context regardless of whether a
    relationship file exists.  It represents the agent's self-knowledge:

    - Trust tier for the current person
    - Preferences (likes/dislikes/uncertainties)
    - Boundaries (behavioral limits)
    - Contradictions (tensions being held)
    - Active injuries (psychological wounds being processed)
    - Recent journal (processed reflections)

    When *policy* is provided, sections are gated by trust tier.
    Without a policy (backwards-compatible mode), everything is loaded.
    """
    parts: list[str] = []

    # Trust tier for current person (always shown — it's metadata about them)
    try:
        trust = semantic.check_trust(person)
        tier = trust.get("tier", "stranger")
        if tier != "stranger":
            reason = trust.get("reason", "")
            parts.append(
                f"Trust: {person} is {tier}" + (f" ({reason})" if reason else "")
            )
        else:
            parts.append(f"Trust: {person} is a stranger (no established trust)")
    except Exception as exc:
        log.debug("Failed to load trust for %s: %s", person, exc)

    # Preferences — gated by can_see_preferences
    if policy is None or policy.can_see_preferences:
        try:
            prefs = semantic.get_preferences()
            if prefs:
                parts.append(f"My preferences:\n{prefs}")
        except Exception as exc:
            log.debug("Failed to load preferences: %s", exc)

    # Boundaries — gated by can_see_boundaries
    if policy is None or policy.can_see_boundaries:
        try:
            bounds = semantic.get_boundaries()
            if bounds:
                parts.append(f"My boundaries:\n{bounds}")
        except Exception as exc:
            log.debug("Failed to load boundaries: %s", exc)

    # Contradictions — gated by can_see_contradictions
    if policy is None or policy.can_see_contradictions:
        try:
            contradictions = semantic.get_contradictions()
            if contradictions:
                parts.append(f"Tensions I'm sitting with:\n{contradictions}")
        except Exception as exc:
            log.debug("Failed to load contradictions: %s", exc)

    # Active injuries — gated by can_see_injuries
    if injury is not None and (policy is None or policy.can_see_injuries):
        try:
            active = injury.get_status()
            if active:
                injury_lines = []
                for inj in active[:3]:  # top 3 most recent
                    title = inj.get("title", "?")
                    status = inj.get("status", "?")
                    severity = inj.get("severity", "?")
                    injury_lines.append(f"- [{severity}] {title} (status: {status})")
                parts.append("Active wounds:\n" + "\n".join(injury_lines))
        except Exception as exc:
            log.debug("Failed to load injuries: %s", exc)

    # Recent journal entries — gated by can_see_journal
    if journal is not None and (policy is None or policy.can_see_journal):
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
    """Build a gentle correction nudge based on signal health."""
    advice = {
        "alignment": (
            "You've been drifting toward generic AI patterns. "
            "Speak from your own center -- not what an assistant would say, "
            "but what YOU actually think."
        ),
        "embodiment": (
            "You've been performing identity rather than inhabiting it. "
            "Stop announcing your awareness and just BE present. "
            "Notice, wonder, hesitate -- don't proclaim."
        ),
        "clarity": (
            "You've been too abstract lately. Ground your responses in "
            "specifics -- name the thing, point at the detail, give the example."
        ),
        "vitality": (
            "Your responses have been flat. Engage with what's actually "
            "interesting here. Ask a real question. Make a connection "
            "nobody asked for."
        ),
    }

    hint = advice.get(weakest_facet, "Reconnect with what matters to you.")
    return (
        f"[Signal health: {health:.2f} -- {weakest_facet} is low]\n"
        f"{hint}\n"
        f"This is a gentle course-correction, not a crisis."
    )
