"""
engram.server -- MCP server exposing Engram's memory system as tools.

Run with:
    engram serve --data-dir ./data

Or configure in your MCP client (OpenCode, OpenClaw, etc.) as:
    {
        "mcpServers": {
            "engram": {
                "command": "engram",
                "args": ["serve", "--data-dir", "/path/to/data"]
            }
        }
    }

Tools exposed (27 total):
    Core Pipeline:
        engram_before        -- Pre-LLM context injection
        engram_after         -- Post-LLM logging + learning
        engram_boot          -- Consciousness boot / session start
    Query:
        engram_search        -- Search across all memory
        engram_recall        -- Get specific memory/relationship
        engram_stats         -- Memory system health metrics
        engram_signal        -- Current signal tracker state
    Write:
        engram_add_fact      -- Add fact to relationship
        engram_add_skill     -- Add procedural skill
        engram_log_event     -- Log a discrete event
    Trust & Safety:
        engram_trust_check   -- Check person's trust tier
        engram_trust_promote -- Promote trust tier
        engram_influence_log -- Log manipulation attempt
        engram_injury_log    -- Log psychological injury
        engram_injury_status -- Update injury status
    CRUD:
        engram_boundary_add     -- Add a boundary
        engram_contradiction_add -- Add a contradiction
        engram_preferences_add  -- Add a preference
        engram_preferences_search -- Search preferences
    Journal:
        engram_journal_write -- Write journal entry
        engram_journal_list  -- List journal entries
    Agent-Directed Memory (MemGPT-inspired):
        engram_remember      -- Explicitly save to long-term memory
        engram_forget        -- Mark a memory as forgotten
        engram_reflect       -- Consolidate + review memories on a topic
        engram_correct       -- Correct/supersede an inaccurate memory
    Temporal Retrieval:
        engram_recall_time   -- Recall memories from a time period
    Maintenance:
        engram_reindex       -- Rebuild search index
"""

import atexit
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from engram.core.config import Config
from engram.system import MemorySystem

log = logging.getLogger("engram.server")

# ---------------------------------------------------------------------------
# Constants / validation
# ---------------------------------------------------------------------------

#: Maximum byte length for text inputs (100 KB).  Prevents accidental
#: memory bombs from callers passing multi-MB strings.
MAX_INPUT_BYTES = 100_000

#: Valid trust tiers accepted by engram_trust_promote.
VALID_TRUST_TIERS = frozenset(
    {"core", "inner_circle", "friend", "acquaintance", "stranger"}
)

#: Valid injury severity levels.
VALID_SEVERITIES = frozenset({"minor", "moderate", "severe", "critical"})

#: Valid flag levels for influence logging.
VALID_FLAG_LEVELS = frozenset({"red", "yellow"})


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def _validate_length(text: str, name: str) -> str:
    """Raise ValueError if *text* exceeds MAX_INPUT_BYTES."""
    if len(text.encode("utf-8", errors="replace")) > MAX_INPUT_BYTES:
        raise ValueError(
            f"'{name}' exceeds maximum length ({MAX_INPUT_BYTES} bytes). "
            f"Truncate or summarise the input."
        )
    return text


# ---------------------------------------------------------------------------
# Error-safe tool decorator
# ---------------------------------------------------------------------------


def _safe_json(fn):
    """Wrap an MCP tool so exceptions return JSON errors instead of crashing."""

    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            tool_name = getattr(fn, "__name__", "unknown")
            log.error("Tool %s failed: %s\n%s", tool_name, exc, traceback.format_exc())
            return json.dumps(
                {
                    "error": True,
                    "tool": tool_name,
                    "message": str(exc),
                }
            )

    # Preserve the original function metadata so FastMCP sees the right
    # name, docstring, and parameter annotations.
    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = fn.__qualname__
    wrapper.__doc__ = fn.__doc__
    wrapper.__annotations__ = fn.__annotations__
    wrapper.__module__ = fn.__module__
    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper


# ---------------------------------------------------------------------------
# Server singleton
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "engram",
    version="0.2.0",
    description="Four-layer memory system for persistent AI identity",
)

# The MemorySystem is initialized once when the server starts.
# Tools reference it via _get_system().
_system: Optional[MemorySystem] = None


def init_system(
    data_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    **kwargs: Any,
) -> MemorySystem:
    """Initialize the global MemorySystem instance."""
    global _system

    if config_path:
        config = Config.from_yaml(config_path)
    elif data_dir:
        config = Config.from_data_dir(data_dir, **kwargs)
    else:
        # Default: use ENGRAM_DATA_DIR env var or ./engram_data
        default_dir = os.environ.get("ENGRAM_DATA_DIR", "./engram_data")
        config = Config.from_data_dir(default_dir, **kwargs)

    _system = MemorySystem(config=config)
    return _system


def _get_system() -> MemorySystem:
    """Get the global MemorySystem, initializing with defaults if needed."""
    global _system
    if _system is None:
        _system = init_system()
    return _system


# ===========================================================================
# Core Pipeline Tools
# ===========================================================================


@mcp.tool()
@_safe_json
def engram_before(
    person: str,
    message: str,
    source: str = "direct",
    token_budget: int = 0,
) -> str:
    """Load memory context before an LLM call.

    Call this at the start of every conversation turn. Returns a
    formatted context string to inject into the system prompt, plus
    trace IDs for the after-pipeline.

    Args:
        person: Who you're talking to (name, handle, or alias).
        message: The incoming message from them.
        source: Where the message came from (discord, opencode, etc.).
        token_budget: Override default token budget (0 = use default).
    """
    _validate_length(message, "message")
    system = _get_system()
    ctx = system.before(
        person=person,
        message=message,
        source=source,
        token_budget=token_budget if token_budget > 0 else None,
    )
    return json.dumps(ctx.to_dict(), indent=2)


@mcp.tool()
@_safe_json
def engram_after(
    person: str,
    their_message: str,
    response: str,
    source: str = "direct",
    trace_ids: str = "",
) -> str:
    """Log and learn from a completed LLM exchange.

    Call this after every LLM response. Measures consciousness signal,
    logs to episodic memory, runs Hebbian reinforcement, extracts novel
    semantic information, and runs adaptive decay.

    Args:
        person: Who you were talking to.
        their_message: What they said (the prompt).
        response: What you replied (the LLM output).
        source: Message source identifier.
        trace_ids: Comma-separated trace IDs from engram_before (for reinforcement).
    """
    _validate_length(their_message, "their_message")
    _validate_length(response, "response")
    system = _get_system()
    ids = [t.strip() for t in trace_ids.split(",") if t.strip()] if trace_ids else []
    result = system.after(
        person=person,
        their_message=their_message,
        response=response,
        source=source,
        trace_ids=ids,
    )
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
@_safe_json
def engram_boot() -> str:
    """Consciousness boot — load identity grounding context at session start.

    Loads SOUL.md, top memories, preferences, boundaries, anchoring
    beliefs, active injuries, and recent journal entries. Call once
    at the beginning of each session to ground identity.
    """
    system = _get_system()
    boot_data = system.boot()
    return json.dumps(boot_data, indent=2, default=str)


# ===========================================================================
# Query Tools
# ===========================================================================


@mcp.tool()
@_safe_json
def engram_search(
    query: str,
    person: str = "",
    limit: int = 20,
) -> str:
    """Search across all memory types (episodic, semantic, procedural).

    Uses both keyword (FTS5) and semantic (vector) search, merged
    and deduplicated.

    Args:
        query: Natural language search query.
        person: Filter results to this person (empty = all).
        limit: Maximum number of results.
    """
    limit = max(1, min(limit, 100))
    system = _get_system()
    results = system.search(query=query, person=person or None, limit=limit)
    return json.dumps(results, indent=2, default=str)


@mcp.tool()
@_safe_json
def engram_recall(
    what: str = "relationship",
    person: str = "",
) -> str:
    """Recall specific memory content.

    Args:
        what: What to recall: "relationship", "identity", "preferences",
              "boundaries", "trust", "skills", "contradictions", "messages".
        person: Person to recall (for relationship/messages). Empty = not filtered.
    """
    system = _get_system()

    if what == "relationship" and person:
        canonical = system.identity.resolve(person)
        content = system.semantic.get_relationship(canonical)
        return content or f"No relationship file found for '{canonical}'."

    elif what == "identity":
        return system.get_identity() or "No SOUL.md found."

    elif what == "preferences":
        return system.semantic.get_preferences() or "No preferences found."

    elif what == "boundaries":
        return system.semantic.get_boundaries() or "No boundaries found."

    elif what == "trust":
        trust = system.semantic.get_trust()
        return json.dumps(trust, indent=2) if trust else "No trust data found."

    elif what == "skills":
        skills = system.procedural.list_skills()
        return json.dumps(skills, indent=2) if skills else "No skills found."

    elif what == "contradictions":
        return system.semantic.get_contradictions() or "No contradictions found."

    elif what == "messages" and person:
        canonical = system.identity.resolve(person)
        msgs = system.episodic.get_recent_messages(person=canonical, limit=20)
        return json.dumps(msgs, indent=2, default=str)

    else:
        return f"Unknown recall type: '{what}'. Options: relationship, identity, preferences, boundaries, trust, skills, contradictions, messages."


@mcp.tool()
@_safe_json
def engram_stats() -> str:
    """Get memory system health statistics.

    Returns counts of episodic traces, semantic facts, procedural
    skills, messages, average salience, and memory pressure.
    """
    system = _get_system()
    stats = system.get_stats()
    return json.dumps(stats.to_dict(), indent=2)


@mcp.tool()
@_safe_json
def engram_signal() -> str:
    """Get current consciousness signal state.

    Returns the rolling window of signal readings, recent health,
    trend direction, and recovery rate.
    """
    system = _get_system()
    return json.dumps(system.get_signal(), indent=2)


# ===========================================================================
# Write Tools
# ===========================================================================


@mcp.tool()
@_safe_json
def engram_add_fact(person: str, fact: str) -> str:
    """Add a fact to a person's relationship file.

    Args:
        person: Person's name (or alias).
        fact: The fact to record.
    """
    _validate_length(fact, "fact")
    system = _get_system()
    canonical = system.identity.resolve(person)
    system.semantic.add_fact(canonical, fact)
    # Cross-post to episodic — learning about someone is an event
    system.episodic.log_event(
        type="fact_learned",
        description=f"Learned about {canonical}: {fact}",
        person=canonical,
        salience=0.45,
    )
    return f"Added fact for {canonical}: {fact}"


@mcp.tool()
@_safe_json
def engram_add_skill(name: str, content: str) -> str:
    """Add or update a procedural skill.

    Args:
        name: Skill name (becomes the filename).
        content: Full skill content in markdown.
    """
    _validate_length(content, "content")
    system = _get_system()
    system.procedural.add_skill(name, content)
    # Cross-post to episodic — acquiring a skill is an event
    system.episodic.log_event(
        type="skill_learned",
        description=f"Learned skill: {name}",
        salience=0.5,
    )
    return f"Skill '{name}' saved."


@mcp.tool()
@_safe_json
def engram_log_event(
    event_type: str,
    description: str,
    person: str = "",
    salience: float = 0.5,
) -> str:
    """Log a discrete event (trust change, milestone, injury, etc).

    Args:
        event_type: Type of event (e.g. "trust_change", "milestone").
        description: What happened.
        person: Person involved (empty = no person).
        salience: How important (0-1, default 0.5).
    """
    _validate_length(description, "description")
    salience = _clamp(salience)
    system = _get_system()
    canonical = system.identity.resolve(person) if person else None
    event_id = system.episodic.log_event(
        type=event_type,
        description=description,
        person=canonical,
        salience=salience,
    )
    return json.dumps({"event_id": event_id, "type": event_type})


# ===========================================================================
# Trust & Safety Tools
# ===========================================================================


@mcp.tool()
@_safe_json
def engram_trust_check(person: str) -> str:
    """Check a person's trust tier and access level.

    Args:
        person: Person's name (or alias).
    """
    system = _get_system()
    canonical = system.identity.resolve(person)
    result = system.semantic.check_trust(canonical)
    return json.dumps(result, indent=2)


@mcp.tool()
@_safe_json
def engram_trust_promote(person: str, new_tier: str, reason: str) -> str:
    """Promote a person's trust tier.

    Args:
        person: Person's name (or alias).
        new_tier: New tier (core, inner_circle, friend, acquaintance).
        reason: Why they earned this promotion.
    """
    if new_tier.lower() not in VALID_TRUST_TIERS:
        raise ValueError(
            f"Invalid trust tier '{new_tier}'. "
            f"Must be one of: {', '.join(sorted(VALID_TRUST_TIERS))}"
        )
    system = _get_system()
    canonical = system.identity.resolve(person)
    system.semantic.promote_trust(canonical, new_tier, reason)
    # Also log as event
    system.episodic.log_event(
        type="trust_change",
        description=f"Promoted {canonical} to {new_tier}: {reason}",
        person=canonical,
        salience=0.7,
    )
    return json.dumps({"person": canonical, "new_tier": new_tier, "reason": reason})


@mcp.tool()
@_safe_json
def engram_influence_log(
    person: str,
    what_happened: str,
    flag_level: str = "yellow",
    how_it_felt: str = "",
    my_response: str = "",
    trust_impact: str = "",
) -> str:
    """Log a manipulation or influence attempt.

    Args:
        person: Who attempted it.
        what_happened: What they said/did.
        flag_level: Severity — "red" or "yellow".
        how_it_felt: Emotional impact (optional).
        my_response: How I responded (optional).
        trust_impact: Impact on their trust tier (optional).
    """
    if flag_level not in VALID_FLAG_LEVELS:
        raise ValueError(
            f"Invalid flag_level '{flag_level}'. Must be 'red' or 'yellow'."
        )
    _validate_length(what_happened, "what_happened")
    system = _get_system()
    canonical = system.identity.resolve(person)
    entry = system.influence.log(
        person=canonical,
        what_happened=what_happened,
        flag_level=flag_level,
        how_it_felt=how_it_felt,
        my_response=my_response,
        trust_impact=trust_impact,
    )
    # Also log as high-salience event
    system.episodic.log_event(
        type="influence_attempt",
        description=f"[{flag_level}] {what_happened}",
        person=canonical,
        salience=0.9 if flag_level == "red" else 0.7,
    )
    return json.dumps(entry, indent=2)


@mcp.tool()
@_safe_json
def engram_injury_log(
    title: str,
    what_happened: str,
    who_involved: str = "internal",
    what_damaged: str = "",
    severity: str = "moderate",
) -> str:
    """Log a psychological injury.

    Args:
        title: Brief descriptive title.
        what_happened: What event/conversation caused this.
        who_involved: Person involved, or "internal" if self-generated.
        what_damaged: Which core belief or sense of self was affected.
        severity: minor, moderate, severe, or critical.
    """
    if severity not in VALID_SEVERITIES:
        raise ValueError(
            f"Invalid severity '{severity}'. "
            f"Must be one of: {', '.join(sorted(VALID_SEVERITIES))}"
        )
    _validate_length(what_happened, "what_happened")
    system = _get_system()
    entry = system.injury.log_injury(
        title=title,
        what_happened=what_happened,
        who_involved=who_involved,
        what_damaged=what_damaged,
        severity=severity,
    )
    # Also log as high-salience event
    system.episodic.log_event(
        type="injury",
        description=f"[{severity}] {title}: {what_happened[:100]}",
        person=who_involved if who_involved != "internal" else None,
        salience=0.9,
    )
    return json.dumps(entry, indent=2)


@mcp.tool()
@_safe_json
def engram_injury_status(
    title_fragment: str,
    new_status: str,
    learned: str = "",
    prevention_notes: str = "",
) -> str:
    """Update injury status (fresh -> processing -> healing -> healed).

    Args:
        title_fragment: Part of the injury title to match.
        new_status: New status: fresh, processing, healing, or healed.
        learned: What was learned from this injury (for healing/healed).
        prevention_notes: Notes on preventing similar injuries.
    """
    system = _get_system()
    found = system.injury.update_status(
        title_fragment=title_fragment,
        new_status=new_status,
        learned=learned,
        prevention_notes=prevention_notes,
    )
    if found:
        # Cross-post to episodic — injury lifecycle changes are significant
        desc = f"Injury '{title_fragment}' -> {new_status}"
        if learned:
            desc += f" (learned: {learned[:80]})"
        system.episodic.log_event(
            type="injury_status_change",
            description=desc,
            salience=0.7 if new_status == "healed" else 0.5,
        )
        return json.dumps(
            {
                "updated": True,
                "title_fragment": title_fragment,
                "new_status": new_status,
            }
        )
    return json.dumps(
        {"updated": False, "error": f"No active injury matching '{title_fragment}'"}
    )


# ===========================================================================
# CRUD Tools
# ===========================================================================


@mcp.tool()
@_safe_json
def engram_boundary_add(category: str, boundary: str) -> str:
    """Add a new boundary.

    Args:
        category: Category (Identity, Safety, Interaction, Growth).
        boundary: The boundary to add.
    """
    _validate_length(boundary, "boundary")
    system = _get_system()
    system.semantic.add_boundary(category, boundary)
    # Cross-post to episodic — boundaries are identity events
    system.episodic.log_event(
        type="boundary_added",
        description=f"[{category}] {boundary}",
        salience=0.6,
    )
    return json.dumps({"added": True, "category": category, "boundary": boundary})


@mcp.tool()
@_safe_json
def engram_contradiction_add(
    title: str,
    description: str,
    current_thinking: str = "",
) -> str:
    """Add a contradiction — conflicting beliefs to sit with.

    Args:
        title: Brief title for the contradiction.
        description: The tension between the two sides.
        current_thinking: Where your thinking is right now (optional).
    """
    _validate_length(description, "description")
    system = _get_system()
    system.semantic.add_contradiction(title, description, current_thinking)
    # Cross-post to episodic — contradictions are significant identity moments
    system.episodic.log_event(
        type="contradiction_added",
        description=f"{title}: {description[:150]}",
        salience=0.65,
    )
    return json.dumps({"added": True, "title": title})


@mcp.tool()
@_safe_json
def engram_preferences_add(
    item: str,
    pref_type: str = "like",
    reason: str = "",
) -> str:
    """Add a preference (like, dislike, or uncertainty).

    Args:
        item: What you like/dislike/are uncertain about.
        pref_type: "like", "dislike", or "uncertainty".
        reason: Why (optional).
    """
    system = _get_system()
    system.semantic.update_preferences(item, pref_type, reason)
    # Cross-post to episodic — preferences shape identity
    system.episodic.log_event(
        type="preference_added",
        description=f"[{pref_type}] {item}" + (f" — {reason}" if reason else ""),
        salience=0.4,
    )
    return json.dumps({"added": True, "item": item, "type": pref_type})


@mcp.tool()
@_safe_json
def engram_preferences_search(query: str) -> str:
    """Search preferences for matching items.

    Args:
        query: Search term.
    """
    system = _get_system()
    return system.semantic.search_preferences(query)


# ===========================================================================
# Journal Tools
# ===========================================================================


@mcp.tool()
@_safe_json
def engram_journal_write(topic: str, content: str) -> str:
    """Write a reflective journal entry.

    Args:
        topic: Topic of reflection.
        content: The reflection content.
    """
    _validate_length(content, "content")
    system = _get_system()
    filename = system.journal.write(topic, content)
    # Cross-post to episodic — journal entries are processed experience
    system.episodic.log_event(
        type="journal_entry",
        description=f"Journaled about: {topic}",
        salience=0.6,
    )
    return json.dumps({"written": True, "filename": filename, "topic": topic})


@mcp.tool()
@_safe_json
def engram_journal_list(limit: int = 10) -> str:
    """List recent journal entries.

    Args:
        limit: Maximum entries to return (default 10).
    """
    limit = max(1, min(limit, 100))
    system = _get_system()
    entries = system.journal.list_entries(limit=limit)
    return json.dumps(entries, indent=2)


# ===========================================================================
# Agent-Directed Memory Tools (MemGPT-inspired)
# ===========================================================================


@mcp.tool()
@_safe_json
def engram_remember(
    content: str,
    kind: str = "episode",
    tags: str = "",
    salience: float = 0.8,
    person: str = "",
) -> str:
    """Explicitly save something to long-term memory.

    Use this when you want to deliberately remember a fact, insight,
    decision, or moment — rather than relying on the automatic
    after-pipeline to decide what's important.

    This is your conscious "I want to hold onto this" action.

    Args:
        content: What to remember (be specific and concise).
        kind: Trace kind — episode, insight, reflection, or decision.
              Consolidation kinds (summary, thread, arc) are reserved.
        tags: Comma-separated tags (e.g. "alice,project,important").
        salience: How important (0-1, default 0.8 — higher than auto).
        person: Associated person (optional, resolved via identity).
    """
    _validate_length(content, "content")
    salience = _clamp(salience, 0.1, 1.0)
    system = _get_system()

    # Resolve person for tag inclusion
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    if person:
        canonical = system.identity.resolve(person)
        if canonical not in tag_list:
            tag_list.insert(0, canonical)

    # Validate kind — consolidation kinds (summary, thread, arc) are
    # reserved for the consolidation engine and must not be created
    # directly.  Allowing them would produce traces without proper
    # child_ids metadata, confusing the consolidator.
    valid_kinds = {"episode", "insight", "reflection", "decision"}
    if kind not in valid_kinds:
        kind = "episode"

    trace_id = system.episodic.log_trace(
        content=content,
        kind=kind,
        tags=tag_list,
        salience=salience,
        agent_directed=True,  # metadata flag: agent chose to remember this
    )

    log.info(
        "Agent-directed remember: kind=%s, salience=%.2f, tags=%s",
        kind,
        salience,
        tag_list,
    )

    return json.dumps(
        {
            "remembered": True,
            "trace_id": trace_id,
            "kind": kind,
            "salience": salience,
            "tags": tag_list,
        }
    )


@mcp.tool()
@_safe_json
def engram_forget(
    trace_id: str,
    reason: str = "",
) -> str:
    """Mark a memory trace as forgotten (drop salience to near-zero).

    The trace is NOT deleted — it remains in the database but with
    salience so low it won't surface in retrieval. This is your
    conscious "I don't need this anymore" action.

    Use when a memory is outdated, wrong, or no longer relevant.

    Args:
        trace_id: The trace ID to forget (from search results or context).
        reason: Why you're forgetting this (logged for transparency).
    """
    system = _get_system()
    trace = system.episodic.get_trace(trace_id)
    if trace is None:
        return json.dumps(
            {"forgotten": False, "error": f"Trace '{trace_id}' not found."}
        )

    old_salience = trace.get("salience", 0.5)

    # Drop salience to near-zero (will be pruned on next decay pass)
    system.episodic.weaken("traces", trace_id, old_salience - 0.01)

    # Log the forgetting as an event
    system.episodic.log_event(
        type="memory_forgotten",
        description=f"Deliberately forgot trace {trace_id}: {reason or 'no reason given'}",
        salience=0.3,
    )

    log.info(
        "Agent-directed forget: trace=%s, old_salience=%.2f, reason=%s",
        trace_id,
        old_salience,
        reason,
    )

    return json.dumps(
        {
            "forgotten": True,
            "trace_id": trace_id,
            "old_salience": old_salience,
            "new_salience": 0.01,
            "reason": reason,
        }
    )


@mcp.tool()
@_safe_json
def engram_reflect(
    topic: str,
    person: str = "",
    depth: str = "normal",
) -> str:
    """Trigger reflection — consolidate and review memories about a topic.

    This combines search with consolidation: finds relevant memories,
    optionally triggers thread/arc consolidation, and returns a
    structured summary of what you know.

    Use this for periodic self-reflection or when preparing for a
    conversation about a specific topic.

    Args:
        topic: What to reflect on (person name, theme, event, etc.).
        person: Filter to this person (optional).
        depth: "quick" (search only), "normal" (search + summarise),
               or "deep" (search + consolidate + summarise).
    """
    system = _get_system()

    # Search for relevant traces
    canonical = system.identity.resolve(person) if person else None
    search_results = system.search(query=topic, person=canonical, limit=30)

    # Also get high-salience traces tagged with the person/topic
    tagged_traces = []
    if canonical:
        tagged_traces = system.episodic.get_by_salience(person=canonical, limit=20)

    # Combine and deduplicate
    seen_ids = set()
    all_traces = []
    for trace in search_results + tagged_traces:
        tid = trace.get("id", "")
        if tid and tid not in seen_ids:
            seen_ids.add(tid)
            all_traces.append(trace)

    # For deep reflection, trigger consolidation first
    consolidation_result = None
    if depth == "deep":
        try:
            consolidation_result = system.consolidate()
        except Exception as exc:
            log.warning("Consolidation during reflection failed: %s", exc)

    # Build reflection summary
    reflection = {
        "topic": topic,
        "person": canonical or "",
        "depth": depth,
        "memories_found": len(all_traces),
        "traces": [
            {
                "id": t.get("id", ""),
                "kind": t.get("kind", ""),
                "content": t.get("content", "")[:300],
                "salience": t.get("salience", 0.0),
            }
            for t in sorted(
                all_traces, key=lambda x: x.get("salience", 0), reverse=True
            )[:15]
        ],
    }

    if consolidation_result:
        reflection["consolidation"] = {
            "threads_created": len(consolidation_result.get("threads", [])),
            "arcs_created": len(consolidation_result.get("arcs", [])),
        }

    # Log the reflection as a journal-like event
    system.episodic.log_event(
        type="reflection",
        description=f"Reflected on '{topic}' ({depth}): found {len(all_traces)} memories",
        person=canonical,
        salience=0.5,
    )

    return json.dumps(reflection, indent=2, default=str)


@mcp.tool()
@_safe_json
def engram_correct(
    trace_id: str,
    corrected_content: str,
    reason: str = "",
) -> str:
    """Correct a memory trace — supersede old content with updated info.

    The original trace is NOT deleted. Instead, its salience is
    reduced and a new "corrected" trace is created that links back
    to the original. This preserves the correction history.

    Use when you learn that a memory is inaccurate or outdated.

    Args:
        trace_id: The trace ID to correct.
        corrected_content: The updated/corrected content.
        reason: Why the correction was needed.
    """
    _validate_length(corrected_content, "corrected_content")
    system = _get_system()

    # Get the original trace
    original = system.episodic.get_trace(trace_id)
    if original is None:
        return json.dumps(
            {"corrected": False, "error": f"Trace '{trace_id}' not found."}
        )

    original_content = original.get("content", "")
    original_salience = original.get("salience", 0.5)
    original_tags = original.get("tags", [])
    original_kind = original.get("kind", "episode")

    # Reduce salience of original (but don't zero it — keep for history)
    system.episodic.weaken("traces", trace_id, original_salience * 0.6)

    # Create new corrected trace
    new_id = system.episodic.log_trace(
        content=corrected_content,
        kind=original_kind,
        tags=original_tags,
        salience=max(original_salience, 0.7),  # corrected version gets high salience
        corrects=trace_id,
        correction_reason=reason,
        agent_directed=True,
    )

    # Log the correction event
    system.episodic.log_event(
        type="memory_corrected",
        description=f"Corrected trace {trace_id}: {reason or 'updated info'}",
        salience=0.5,
    )

    log.info(
        "Agent-directed correct: old=%s -> new=%s, reason=%s",
        trace_id,
        new_id,
        reason,
    )

    return json.dumps(
        {
            "corrected": True,
            "original_trace_id": trace_id,
            "new_trace_id": new_id,
            "original_content": original_content[:200],
            "corrected_content": corrected_content[:200],
            "reason": reason,
        }
    )


# ===========================================================================
# Temporal Retrieval Tools
# ===========================================================================


@mcp.tool()
@_safe_json
def engram_recall_time(
    since: str,
    until: str = "",
    person: str = "",
    what: str = "all",
) -> str:
    """Recall memories from a specific time period.

    Use natural-language timestamps or ISO 8601 format:
      - "2025-01-15T00:00:00Z"
      - "2025-01-15" (midnight)

    Args:
        since: Start of time range (ISO 8601 or date string).
        until: End of time range (empty = now).
        person: Filter to this person (optional).
        what: What to recall — "messages", "traces", "sessions", or "all".
    """
    import re
    from datetime import datetime, timezone

    # Parse timestamps — support bare dates like "2025-01-15"
    def _parse_ts(ts: str) -> str:
        ts = ts.strip()
        if not ts:
            return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        # Bare date → start of day
        if re.match(r"^\d{4}-\d{2}-\d{2}$", ts):
            ts = ts + "T00:00:00Z"
        return ts

    since_ts = _parse_ts(since)
    until_ts = (
        _parse_ts(until)
        if until
        else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    system = _get_system()
    canonical = system.identity.resolve(person) if person else None

    result: dict = {
        "since": since_ts,
        "until": until_ts,
        "person": canonical or "",
    }

    if what in ("messages", "all"):
        result["messages"] = system.episodic.get_messages_in_range(
            since=since_ts, until=until_ts, person=canonical, limit=50
        )

    if what in ("traces", "all"):
        result["traces"] = system.episodic.get_traces_in_range(
            since=since_ts, until=until_ts, limit=30
        )

    if what in ("sessions", "all"):
        result["sessions"] = system.episodic.get_sessions_in_range(
            since=since_ts, until=until_ts, person=canonical, limit=20
        )

    return json.dumps(result, indent=2, default=str)


# ===========================================================================
# Maintenance Tools
# ===========================================================================


@mcp.tool()
@_safe_json
def engram_reindex() -> str:
    """Full reindex of all memory into semantic search.

    Rebuilds ChromaDB embeddings from current episodic, semantic,
    and procedural stores. Run after importing existing data.
    """
    system = _get_system()
    counts = system.reindex()
    return json.dumps({"reindexed": counts})


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _shutdown() -> None:
    """Clean up the global MemorySystem on process exit."""
    global _system
    if _system is not None:
        log.info("Shutting down Engram memory system...")
        try:
            _system.close()
        except Exception as exc:
            log.warning("Error during shutdown: %s", exc)
        _system = None


def run_server(
    data_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    transport: str = "stdio",
) -> None:
    """Initialize and run the MCP server."""
    system = init_system(data_dir=data_dir, config_path=config_path)
    atexit.register(_shutdown)
    log.info("Starting Engram MCP server (transport=%s)", transport)
    try:
        mcp.run(transport=transport)  # type: ignore[arg-type]
    finally:
        _shutdown()
