"""
engram.pipeline.after -- Post-LLM logging and learning pipeline.

Called automatically after every LLM response.  Performs:
  1. Measure code quality signal (hybrid: regex + LLM judge)
     + track signal in rolling window
  2. Derive salience from signal health
  3. Session boundary detection + auto-management
  4. Log the exchange to episodic memory (both messages + trace)
  5. Hebbian reinforcement on context traces
  6. Semantic extraction + apply updates (LLM-based, optional)
  7. Pressure-aware decay, compaction, and consolidation

Everything here is fire-and-forget -- failures in any step are
logged and swallowed so the user never sees memory-system errors.
"""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Dict, List, Optional

from engram.core.types import AfterResult, LLMFunc, Signal

# Shared thread pool for parallelising independent LLM calls in the
# after-pipeline.  max_workers=2 because we only parallelise signal
# measurement and semantic extraction (the two LLM-bound steps).
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="engram-after")

if TYPE_CHECKING:
    from engram.consolidation.compactor import ConversationCompactor
    from engram.consolidation.consolidator import MemoryConsolidator
    from engram.consolidation.pressure import MemoryPressure
    from engram.core.config import Config
    from engram.episodic.store import EpisodicStore
    from engram.procedural.store import ProceduralStore
    from engram.semantic.store import SemanticStore
    from engram.signal.decay import DecayEngine
    from engram.signal.measure import SignalTracker
    from engram.signal.reinforcement import ReinforcementEngine
    from engram.workspace import CognitiveWorkspace

log = logging.getLogger("engram.pipeline.after")

# Pipeline-level dedup: tracks recent exchange hashes to prevent the
# entire after-pipeline from running twice on the same exchange (e.g.
# when Discord fires messageCreate twice or both SKILL.md and an
# automatic hook call engram_after).
_RECENT_EXCHANGE_HASHES: Dict[str, float] = {}
_DEDUP_WINDOW_SECONDS: float = 30.0
_MAX_CACHE_SIZE: int = 100


def _exchange_hash(person: str, their_message: str, response: str) -> str:
    """Compute a short hash of the exchange for dedup purposes."""
    raw = f"{person}|{their_message}|{response}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _is_duplicate_exchange(person: str, their_message: str, response: str) -> bool:
    """Check if this exact exchange was already processed recently."""
    now = time.time()
    h = _exchange_hash(person, their_message, response)

    # Evict stale entries
    stale = [
        k for k, t in _RECENT_EXCHANGE_HASHES.items() if now - t > _DEDUP_WINDOW_SECONDS
    ]
    for k in stale:
        del _RECENT_EXCHANGE_HASHES[k]

    # Safety: cap cache size
    if len(_RECENT_EXCHANGE_HASHES) > _MAX_CACHE_SIZE:
        _RECENT_EXCHANGE_HASHES.clear()

    if h in _RECENT_EXCHANGE_HASHES:
        log.info("Duplicate exchange detected (hash=%s), skipping after-pipeline", h)
        return True

    _RECENT_EXCHANGE_HASHES[h] = now
    return False


def after(
    *,
    person: str,
    their_message: str,
    response: str,
    source: str = "direct",
    trace_ids: Optional[List[str]] = None,
    config: "Config",
    episodic: "EpisodicStore",
    semantic: "SemanticStore",
    procedural: "ProceduralStore",
    reinforcement: "ReinforcementEngine",
    decay_engine: "DecayEngine",
    signal_tracker: "SignalTracker",
    llm_func: Optional[LLMFunc] = None,
    memory_pressure: Optional["MemoryPressure"] = None,
    compactor: Optional["ConversationCompactor"] = None,
    consolidator: Optional["MemoryConsolidator"] = None,
    skip_persistence: bool = False,
    workspace: Optional["CognitiveWorkspace"] = None,
) -> AfterResult:
    """Run the full after-pipeline and return results.

    Parameters
    ----------
    person:
        Canonical name of the conversation partner.
    their_message:
        What they said (the prompt).
    response:
        What we replied (the LLM output to evaluate).
    source:
        Message source identifier.
    trace_ids:
        Trace IDs that were loaded into context (from before-pipeline).
    config:
        Engram configuration.
    episodic:
        Episodic memory store.
    semantic:
        Semantic memory store.
    procedural:
        Procedural memory store.
    reinforcement:
        Hebbian reinforcement engine.
    decay_engine:
        Adaptive decay engine.
    signal_tracker:
        Rolling signal window tracker.
    llm_func:
        Optional LLM callable for signal measurement and extraction.
    memory_pressure:
        Optional MemoryPressure monitor. When provided, decay and
        compaction are throttled based on pressure level instead of
        running unconditionally.
    compactor:
        Optional ConversationCompactor. Runs when memory pressure
        is elevated or critical.
    consolidator:
        Optional MemoryConsolidator. Runs hierarchical consolidation
        (episodes -> threads -> arcs) when memory pressure is elevated
        or critical.
    skip_persistence:
        When True, signal is still measured (for system health) but
        nothing is written to episodic memory.
    workspace:
        Optional cognitive workspace.  When provided, ``age_step()``
        is called to decay workspace items after each exchange.

    Returns
    -------
    AfterResult
        Signal reading, salience, updates applied, logged IDs.
    """
    # -- 0. Pipeline-level dedup guard ------------------------------------
    #    If this exact exchange (person + their_message + response) was
    #    already processed within the last 30s, return a minimal result
    #    instead of double-logging everything.
    if _is_duplicate_exchange(person, their_message, response):
        return AfterResult(
            signal=Signal(trace_ids=trace_ids or []),
            salience=0.5,
            updates=[],
            logged_message_id="",
            logged_trace_id=None,
            timings={"total": 0.0, "skipped_dedup": 1.0},
        )

    trace_ids = trace_ids or []
    updates: List[Dict] = []
    timings: Dict[str, float] = {}
    _t0_total = time.perf_counter()

    # -- 0b. Launch extraction in background --------------------------------
    #    Semantic extraction (LLM-based) is independent of signal
    #    measurement.  By launching it in a background thread, both
    #    LLM calls run concurrently: total latency drops from
    #    ~signal_ms + ~extraction_ms to ~max(signal_ms, extraction_ms).
    extraction_future: Optional[Future] = None
    should_extract = (
        not skip_persistence and config.extract_mode == "llm" and llm_func is not None
    )
    if should_extract:
        extraction_future = _EXECUTOR.submit(
            _extract_and_apply,
            person=person,
            their_message=their_message,
            response=response,
            semantic=semantic,
            procedural=procedural,
            llm_func=llm_func,
        )

    # -- 1. Measure code quality signal ------------------------------------
    _t0 = time.perf_counter()
    signal = _measure_signal(
        response=response,
        config=config,
        semantic=semantic,
        their_message=their_message,
        trace_ids=trace_ids,
        llm_func=llm_func,
    )
    timings["signal_measure"] = time.perf_counter() - _t0

    # -- 1b. Style adherence check (adjusts consistency facet) -------------
    #    If the response contains code, run the style drift detector and
    #    blend the result into the consistency facet.  This creates a
    #    tighter feedback loop: style drift -> lower CQS -> lower salience.
    _t0 = time.perf_counter()
    signal = _apply_style_check(signal, response)
    timings["style_check"] = time.perf_counter() - _t0

    signal_tracker.record(signal)

    # -- 2. Derive salience from signal health -----------------------------
    salience = _derive_salience(signal.health, their_message, response)

    # -- 3-7: Persistence steps (skipped for strangers) --------------------
    logged_msg_id: str = ""
    logged_trace_id: Optional[str] = None

    if skip_persistence:
        log.debug("Skipping persistence for %s (memory_persistent=False)", person)
    else:
        # -- 3-6: Batched writes -------------------------------------------
        #    All SQLite writes (log_message, log_trace, reinforce, weaken,
        #    update_access) are batched into a single commit.  Under WAL
        #    mode this reduces disk I/O by 5-10x.
        with episodic.batch():
            # -- 3. Session boundary detection ----------------------------
            _t0 = time.perf_counter()
            _manage_session(person=person, episodic=episodic)
            timings["session_mgmt"] = time.perf_counter() - _t0

            # -- 4. Log exchange to episodic memory -----------------------
            _t0 = time.perf_counter()
            logged_msg_id, logged_trace_id = _log_exchange(
                person=person,
                their_message=their_message,
                response=response,
                source=source,
                salience=salience,
                signal=signal,
                episodic=episodic,
            )
            timings["log_exchange"] = time.perf_counter() - _t0

            # -- 5. Hebbian reinforcement on context traces ---------------
            _t0 = time.perf_counter()
            _reinforce(
                trace_ids=trace_ids,
                signal_health=signal.health,
                reinforcement=reinforcement,
                episodic=episodic,
                response=response,
            )
            timings["reinforcement"] = time.perf_counter() - _t0

        # -- 6. Collect extraction results (launched in step 0b) -----------
        if extraction_future is not None:
            _t0 = time.perf_counter()
            try:
                extraction_updates = extraction_future.result(timeout=30.0)
                updates.extend(extraction_updates)
            except Exception as exc:
                log.warning("Parallel extraction failed: %s", exc)
            timings["extraction"] = time.perf_counter() - _t0

        # -- 7. Pressure-aware decay + compaction (MemGPT-inspired) --------
        _t0 = time.perf_counter()
        _run_maintenance(
            person=person,
            episodic=episodic,
            decay_engine=decay_engine,
            signal_tracker=signal_tracker,
            config=config,
            memory_pressure=memory_pressure,
            compactor=compactor,
            consolidator=consolidator,
            llm_func=llm_func,
        )
        timings["maintenance"] = time.perf_counter() - _t0

    # -- 8. Workspace age step — decay working memory priorities -----------
    if workspace is not None:
        try:
            expired = workspace.age_step()
            if expired > 0:
                log.debug("Workspace: %d items expired", expired)
        except Exception as exc:
            log.debug("Workspace age_step failed: %s", exc)

    timings["total"] = time.perf_counter() - _t0_total

    log.info(
        "After pipeline: person=%s, signal=%s (%.2f), salience=%.2f, "
        "updates=%d, msg=%s, total_ms=%.1f",
        person,
        signal.state,
        signal.health,
        salience,
        len(updates),
        logged_msg_id,
        timings["total"] * 1000,
    )

    return AfterResult(
        signal=signal,
        salience=salience,
        updates=updates,
        logged_message_id=logged_msg_id or "",
        logged_trace_id=logged_trace_id,
        timings=timings,
    )


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------


def _manage_session(
    *,
    person: str,
    episodic: "EpisodicStore",
    gap_hours: float = 2.0,
) -> None:
    """Detect session boundaries and auto-manage sessions.

    If there's no active session for this person, or if more than
    ``gap_hours`` have passed since the last message, end the old
    session and start a new one.
    """
    try:
        active = episodic.get_active_session(person)
        is_new = episodic.detect_session_boundary(person, gap_hours=gap_hours)

        if is_new:
            # End the old session if there is one
            if active:
                episodic.end_session(active["id"])
                log.debug(
                    "Ended session %s for %s (%d messages)",
                    active["id"],
                    person,
                    active.get("message_count", 0),
                )

            # Start a new session
            session_id = episodic.start_session(person)
            log.debug("Started new session %s for %s", session_id, person)
        elif active:
            # Increment message count on the active session
            episodic.increment_session_message_count(active["id"])
    except Exception as exc:
        log.debug("Session management failed: %s", exc)


def _measure_signal(
    *,
    response: str,
    config: "Config",
    semantic: "SemanticStore",
    their_message: str,
    trace_ids: List[str],
    llm_func: Optional[LLMFunc],
) -> Signal:
    """Measure code quality signal using hybrid mode."""
    from engram.signal.measure import measure

    # Always attempt hybrid (regex + LLM).  measure() handles fallback.
    use_llm = llm_func if config.signal_mode in ("hybrid", "llm") else None

    soul_text = ""
    try:
        soul_text = semantic.get_identity()
    except Exception:
        pass

    try:
        return measure(
            text=response,
            llm_func=use_llm,
            soul_text=soul_text,
            prompt=their_message,
            trace_ids=trace_ids,
            llm_weight=config.llm_weight,
        )
    except Exception as exc:
        log.warning("Signal measurement failed, using defaults: %s", exc)
        return Signal(trace_ids=trace_ids)


def _apply_style_check(signal: Signal, response: str) -> Signal:
    """Run style drift detection and blend into the consistency facet.

    Only runs if the response appears to contain code (has indented
    lines or code block markers).  The style score is blended with the
    existing consistency facet at a 30/70 ratio (30% style, 70% regex).
    """
    # Quick heuristic: does the response contain code?
    lines = response.split("\n")
    code_indicators = sum(
        1
        for line in lines
        if line.startswith("    ")
        or line.startswith("\t")
        or line.strip().startswith("def ")
        or line.strip().startswith("class ")
        or line.strip().startswith("function ")
        or "```" in line
    )
    if code_indicators < 2:
        return signal  # No meaningful code to assess

    try:
        from engram.signal.style import assess_style, StyleProfile

        # Extract code blocks if wrapped in markdown fences
        code_text = response
        if "```" in response:
            blocks = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    blocks.append(line)
            if blocks:
                code_text = "\n".join(blocks)

        result = assess_style(code_text, StyleProfile.python_default())

        # Blend style score into consistency: 30% style, 70% existing
        blended_consistency = signal.consistency * 0.7 + result.score * 0.3
        return Signal(
            correctness=signal.correctness,
            consistency=blended_consistency,
            completeness=signal.completeness,
            robustness=signal.robustness,
            trace_ids=signal.trace_ids,
        )
    except Exception as exc:
        log.debug("Style check failed: %s", exc)
        return signal


def _derive_salience(health: float, their_message: str, response: str) -> float:
    """Derive salience from signal health and content heuristics.

    Base salience comes from signal health.  Short exchanges get a
    slight penalty; very long substantive exchanges get a boost.
    """
    # Base: signal health maps directly to salience
    base = health

    # Length heuristic: very short exchanges are less likely to be memorable
    total_len = len(their_message) + len(response)
    if total_len < 50:
        base *= 0.7
    elif total_len > 2000:
        base = min(1.0, base * 1.15)

    return max(0.05, min(1.0, base))


def _log_exchange(
    *,
    person: str,
    their_message: str,
    response: str,
    source: str,
    salience: float,
    signal: Signal,
    episodic: "EpisodicStore",
) -> tuple:
    """Log both sides of the exchange to episodic memory.

    Returns (message_id, trace_id).  trace_id may be None if trace
    logging fails.
    """
    msg_id = ""
    trace_id = None

    # Log their message
    try:
        episodic.log_message(
            person=person,
            speaker=person,
            content=their_message,
            source=source,
            salience=salience * 0.8,  # slightly lower salience for input
        )
    except Exception as exc:
        log.warning("Failed to log incoming message: %s", exc)

    # Log our response (this is the primary logged message)
    try:
        msg_id = episodic.log_message(
            person=person,
            speaker="self",
            content=response,
            source=source,
            salience=salience,
            signal=signal.to_dict(),
        )
    except Exception as exc:
        log.warning("Failed to log response: %s", exc)

    # Log a trace summarizing the exchange
    try:
        trace_content = f"Exchange with {person}: they said '{their_message[:200]}', I replied '{response[:200]}'"
        trace_id = episodic.log_trace(
            content=trace_content,
            kind="episode",
            tags=[person, source],
            salience=salience,
        )
    except Exception as exc:
        log.warning("Failed to log trace: %s", exc)

    return msg_id, trace_id


def _reinforce(
    *,
    trace_ids: List[str],
    signal_health: float,
    reinforcement: "ReinforcementEngine",
    episodic: "EpisodicStore",
    response: str = "",
) -> None:
    """Run citation-primary Hebbian reinforcement on context traces.

    Delegates to ReinforcementEngine.process() which differentiates
    between cited and uncited traces:
    - Cited traces get full reinforcement (proven useful by citation)
    - Uncited traces get minimal reinforcement on high signal
    - Only uncited traces are weakened on low signal
    - Cited traces are never weakened (citation protects them)
    """
    if not trace_ids:
        return

    try:
        # Extract which traces were cited in the response
        cited_list = _extract_cited_trace_ids(response, trace_ids)
        cited_set = set(cited_list)

        if cited_set:
            log.debug(
                "Citations detected: %d/%d traces cited",
                len(cited_set),
                len(trace_ids),
            )

        # Citation-primary reinforcement: cited_ids drives differential
        # treatment within the reinforcement engine.
        reinforcement.process(
            trace_ids=trace_ids,
            signal_health=signal_health,
            episodic_store=episodic,
            cited_ids=cited_set,
        )
    except Exception as exc:
        log.warning("Reinforcement failed: %s", exc)


def _extract_cited_trace_ids(response: str, trace_ids: List[str]) -> List[str]:
    """Extract citation numbers from the response and map to trace IDs.

    The context builder numbers traces as [1], [2], etc. in order.
    This function finds those references in the response text and
    returns the corresponding trace IDs.
    """
    import re

    if not response or not trace_ids:
        return []

    # Find all [N] patterns where N is a positive integer
    cited_numbers = set()
    for match in re.finditer(r"\[(\d+)\]", response):
        num = int(match.group(1))
        if 1 <= num <= len(trace_ids):
            cited_numbers.add(num)

    # Map 1-indexed citation numbers to trace IDs
    return [trace_ids[n - 1] for n in sorted(cited_numbers)]


def _extract_and_apply(
    *,
    person: str,
    their_message: str,
    response: str,
    semantic: "SemanticStore",
    procedural: "ProceduralStore",
    llm_func: LLMFunc,
) -> List[Dict]:
    """Run LLM extraction and apply updates to semantic stores."""
    from engram.signal.extract import extract

    updates: List[Dict] = []

    # Gather existing knowledge for extraction context
    existing = semantic.get_relationship(person) or ""

    try:
        extraction = extract(
            person=person,
            their_message=their_message,
            response=response,
            existing_knowledge=existing,
            llm_func=llm_func,
        )
    except Exception as exc:
        log.warning("Semantic extraction failed: %s", exc)
        return []

    if extraction.get("nothing_new"):
        return []

    # -- Apply relationship updates ----------------------------------------
    for update in extraction.get("relationship_updates", []):
        try:
            target = update.get("person", person)
            fact = update.get("fact", "")
            section = update.get("section", "What I Know")
            if fact:
                semantic.add_fact(target, fact)
                updates.append({"type": "relationship", "person": target, "fact": fact})
                log.debug("Added fact for %s: %s", target, fact[:80])
        except Exception as exc:
            log.warning("Failed to apply relationship update: %s", exc)

    # -- Apply preference updates ------------------------------------------
    for update in extraction.get("preference_updates", []):
        try:
            item = update.get("item", "")
            pref_type = update.get("type", "like")
            reason = update.get("reason", "")
            if item:
                semantic.update_preferences(item, pref_type, reason)
                updates.append(
                    {"type": "preference", "item": item, "pref_type": pref_type}
                )
                log.debug("Added preference: %s %s", pref_type, item[:80])
        except Exception as exc:
            log.warning("Failed to apply preference update: %s", exc)

    # -- Apply trust changes -----------------------------------------------
    for update in extraction.get("trust_changes", []):
        try:
            target = update.get("person", person)
            direction = update.get("direction", "")
            reason = update.get("reason", "")
            if direction and reason:
                # We don't auto-change trust tiers, but we log the signal
                updates.append(
                    {
                        "type": "trust_signal",
                        "person": target,
                        "direction": direction,
                        "reason": reason,
                    }
                )
                log.debug(
                    "Trust signal for %s: %s (%s)", target, direction, reason[:80]
                )
        except Exception as exc:
            log.warning("Failed to process trust change: %s", exc)

    # -- Apply skills learned ----------------------------------------------
    for update in extraction.get("skills_learned", []):
        try:
            skill_name = update.get("skill", "")
            content = update.get("content", "")
            if skill_name and content:
                procedural.add_skill(skill_name, content)
                updates.append({"type": "skill", "skill": skill_name})
                log.debug("Added skill: %s", skill_name)
        except Exception as exc:
            log.warning("Failed to add skill: %s", exc)

    return updates


def _run_maintenance(
    *,
    person: str,
    episodic: "EpisodicStore",
    decay_engine: "DecayEngine",
    signal_tracker: "SignalTracker",
    config: "Config",
    memory_pressure: Optional["MemoryPressure"],
    compactor: Optional["ConversationCompactor"],
    consolidator: Optional["MemoryConsolidator"] = None,
    llm_func: Optional[LLMFunc],
) -> None:
    """Run pressure-aware decay, pruning, compaction, and consolidation.

    When a ``MemoryPressure`` monitor is provided, decay is throttled
    based on utilisation level instead of running on every call:

      - NORMAL: decay at most once per hour.
      - ELEVATED: every 10 minutes, plus trigger compaction + consolidation.
      - CRITICAL: every call, plus aggressive compaction + consolidation.

    Without a pressure monitor, falls back to the original unconditional
    decay (backwards compatible).
    """
    try:
        coherence = signal_tracker.recent_health()
        decay_engine.update_coherence(coherence)

        if memory_pressure is not None:
            # Pressure-aware path
            state = memory_pressure.check(episodic)

            if state.should_decay:
                episodic.decay_pass(
                    half_life_hours=config.decay_half_life_hours,
                    coherence=coherence,
                )
                episodic.prune(min_salience=decay_engine.min_salience)
                memory_pressure.record_decay()

            if state.should_compact and compactor is not None:
                try:
                    compactor.compact(person, episodic, llm_func)
                    memory_pressure.record_compaction()
                except Exception as exc:
                    log.warning("Compaction failed for %s: %s", person, exc)

                # Consolidation piggybacks on compaction timing —
                # only when pressure triggers compaction do we also
                # consolidate episodes → threads → arcs.
                if consolidator is not None:
                    try:
                        consolidator.consolidate(episodic, llm_func)
                    except Exception as exc:
                        log.warning("Consolidation failed: %s", exc)
        else:
            # Legacy path: unconditional decay (backwards compatible)
            episodic.decay_pass(
                half_life_hours=config.decay_half_life_hours,
                coherence=coherence,
            )
            episodic.prune(min_salience=decay_engine.min_salience)

    except Exception as exc:
        log.warning("Maintenance pass failed: %s", exc)
