"""
Engram -- MCP server exposing Engram's memory system as tools.

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

Tools exposed (24 total):
    Core Pipeline:
        engram_before        -- Pre-LLM context injection
        engram_after         -- Post-LLM logging + learning
        engram_boot          -- Session start / context priming
    Query:
        engram_search        -- Search across all memory
        engram_recall        -- Get specific memory/relationship
        engram_stats         -- Memory system health metrics
        engram_signal        -- Current signal tracker state
    Write:
        engram_add_fact      -- Add fact to relationship
        engram_add_skill     -- Add procedural skill
        engram_log_event     -- Log a discrete event
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
    Workspace:
        engram_workspace_add      -- Add to working memory
        engram_workspace_status   -- Get working memory status
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
    "Engram",
    version="0.2.0",
    description="Four-layer memory system for persistent AI identity",
)

# The MemorySystem is initialized once when the server starts.
# Tools reference it via _get_system().
_system: Optional[MemorySystem] = None

# Current source/person set by engram_before() for trust gating
# across subsequent tool calls within the same conversation turn.
_current_source: str = "direct"
_current_person: str = ""


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


def _gate_tool(tool_name: str) -> None:
    """Tool access gating stub.

    Trust-based gating has been removed in the code-first pivot.
    This function is kept as a no-op placeholder so existing tool
    decorators don't need to be changed.
    """
    pass


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
    global _current_source, _current_person
    _validate_length(message, "message")
    system = _get_system()

    # Track source/person for trust gating of subsequent tool calls
    _current_source = source
    _current_person = system.identity.resolve(person)

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
    _gate_tool("engram_add_fact")
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
    _gate_tool("engram_add_skill")
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
    _gate_tool("engram_log_event")
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
    _gate_tool("engram_boundary_add")
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
    _gate_tool("engram_contradiction_add")
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
    _gate_tool("engram_preferences_add")
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
    _gate_tool("engram_journal_write")
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
    _gate_tool("engram_remember")
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
    valid_kinds = {
        "episode",
        "realization",
        "reflection",
        "emotion",
        "factual",
        "identity_core",
        "uncertainty",
        "anticipation",
        "creative_journey",
        "emotional_thread",
        "promise",
        "confidence",
    }
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
    _gate_tool("engram_forget")
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
    _gate_tool("engram_correct")
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
# Workspace Tools
# ===========================================================================


@mcp.tool()
@_safe_json
def engram_workspace_add(
    item: str,
    priority: float = 0.5,
    source: str = "manual",
) -> str:
    """Add something to working memory (7±2 capacity).

    If already present, rehearses it instead. If full, evicts the
    lowest-priority item (which is saved to episodic memory).

    Args:
        item: What to hold in working memory.
        priority: How important (0-1).
        source: Where this came from.
    """
    _gate_tool("engram_workspace_add")
    _validate_length(item, "item")
    system = _get_system()
    result = system.workspace.add(
        item=item,
        priority=_clamp(priority),
        source=source,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
@_safe_json
def engram_workspace_status() -> str:
    """Get current working memory status.

    Shows all active slots with their priority, age, and focus state.
    """
    system = _get_system()
    return system.workspace.status()


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
    _gate_tool("engram_reindex")
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
