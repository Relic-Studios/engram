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

Tools exposed (35 total):
    Core Pipeline:
        engram_before        -- Pre-LLM context injection
        engram_after         -- Post-LLM logging + learning
        engram_boot          -- Session start / context priming
    Query:
        engram_search        -- Search across all memory
        engram_recall        -- Get specific memory/relationship
        engram_stats         -- Memory system health metrics
        engram_signal        -- Current code quality signal state
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
    Code-First (Phase 4):
        engram_architecture_decision -- Record an ADR
        engram_code_pattern  -- Store/get reusable code patterns
        engram_debug_log     -- Log error + resolution (Table 3 schema)
        engram_debug_recall  -- Recall prior debug sessions by fingerprint
        engram_get_rules     -- Get coding standards
        engram_add_wiring    -- Record code dependency relationship
        engram_project_init  -- Initialize project scope
    AST / Code Analysis:
        engram_extract_symbols -- AST analysis + optional symbol storage
        engram_repo_map      -- Generate Aider-style repo map
    Multi-Agent:
        engram_sessions      -- List active client sessions
        engram_register_client -- Register/identify an MCP client session
    Maintenance:
        engram_reindex       -- Rebuild search index
"""

import atexit
import json
import logging
import os
import traceback
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from engram.core.config import Config
from engram.core.sessions import (
    ClientSession,
    SessionRegistry,
    get_current_session_id,
    set_current_session_id,
)
from engram.system import MemorySystem
from engram.working.allocator import reorder_u

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

# Session registry: tracks per-client state for multi-agent support.
# In stdio mode (single client), only the default session is used.
# In HTTP/SSE mode, each transport session gets its own ClientSession.
_sessions = SessionRegistry()

# Legacy globals — kept for backward compatibility with tests that
# directly set server_mod._current_source = "direct" etc.
# In production, tools use _get_session() which reads from the
# session registry.  These globals are synced in engram_before()
# and engram_project_init() as a fallback.
_current_source: str = "direct"
_current_person: str = ""
_current_project: str = ""


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


def _get_session() -> ClientSession:
    """Get the ClientSession for the current execution context.

    Uses the contextvars-based session ID to look up the active
    session in the registry.  In stdio mode this always returns
    the default session.  In HTTP/SSE mode, the session ID is set
    per-request by the transport layer.
    """
    return _sessions.get(get_current_session_id())


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

    # Track source/person in both the session registry and legacy globals
    resolved_person = system.identity.resolve(person)
    session = _get_session()
    session.source = source
    session.person = resolved_person
    _current_source = source
    _current_person = resolved_person

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

    Call this after every LLM response. Measures code quality signal,
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
    """Session boot — load project context at session start.

    Loads SOUL.md (coding philosophy), top high-salience memories,
    coding style preferences, and recent journal entries. Call once
    at the beginning of each session to ground the coding context.
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
    # Primacy-recency reordering: place highest-relevance results at
    # the start and end of the list so the LLM attends to them most.
    results = reorder_u(results, key_field="combined_score")
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
    """Get current code quality signal state.

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
        "factual",
        "code_pattern",
        "debug_session",
        "architecture_decision",
        "wiring_map",
        "error_resolution",
        "test_strategy",
        "project_context",
        "code_review",
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
    top_reflect = sorted(all_traces, key=lambda x: x.get("salience", 0), reverse=True)[
        :15
    ]
    # Primacy-recency reordering for LLM attention
    top_reflect = reorder_u(top_reflect, key_field="salience")
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
            for t in top_reflect
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
# Code-First Tools (Phase 4 pivot)
# ===========================================================================


@mcp.tool()
@_safe_json
def engram_architecture_decision(
    context: str,
    options: str,
    decision: str,
    consequences: str = "",
) -> str:
    """Record an Architecture Decision Record (ADR) into persistent memory.

    ADRs capture significant design choices with their rationale and
    alternatives.  They are assigned high initial salience so they are
    always present in the context window during relevant tasks, preventing
    the agent from violating long-term project policies.

    Uses the Michael Nygard ADR format: Context, Options, Decision,
    Consequences.

    Args:
        context: Why is this decision being made? What forces are at play?
        options: What alternatives were considered? (comma-separated or prose)
        decision: What was decided and why?
        consequences: What are the trade-offs? (optional)
    """
    system = _get_system()

    adr_content = (
        f"## Architecture Decision\n\n"
        f"### Context\n{context}\n\n"
        f"### Options Considered\n{options}\n\n"
        f"### Decision\n{decision}\n\n"
    )
    if consequences:
        adr_content += f"### Consequences\n{consequences}\n"

    trace_id = system.episodic.log_trace(
        content=adr_content,
        kind="architecture_decision",
        tags=["adr"],
        salience=0.95,  # High salience — always surfaced at boot
    )
    return json.dumps(
        {
            "trace_id": trace_id,
            "kind": "architecture_decision",
            "salience": 0.95,
            "message": "ADR recorded with high salience — will be surfaced at boot.",
        }
    )


@mcp.tool()
@_safe_json
def engram_code_pattern(
    action: str,
    name: str,
    content: str = "",
    tags: str = "",
    language: str = "",
    framework: str = "",
    category: str = "",
) -> str:
    """Store or retrieve validated code patterns in procedural memory.

    Code patterns are reusable implementation templates that the agent
    can adapt to different contexts.  Patterns are stored as procedural
    skills and also logged as episodic traces for retrieval.

    Args:
        action: "store" to save a pattern, "get" to retrieve one.
        name: Pattern name (e.g., "retry-with-backoff", "repository-pattern").
        content: The pattern code/description (required for "store").
        tags: Comma-separated tags for categorization (e.g., "python,async,error-handling").
        language: Programming language (e.g., "python", "typescript").
        framework: Framework or library (e.g., "asyncio", "react", "fastapi").
        category: Pattern category (e.g., "error-handling", "testing", "api-design").
    """
    system = _get_system()

    if action == "store":
        if not content:
            return json.dumps({"error": "content is required for action='store'"})

        # Parse tags
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        # Store as structured procedural skill with YAML frontmatter
        meta = system.procedural.add_structured_skill(
            name=name,
            content=f"# {name}\n\n{content}",
            language=language,
            framework=framework,
            category=category,
            tags=tag_list,
        )

        # Also log as episodic trace for search/retrieval
        ep_tags = list(tag_list)
        if language:
            ep_tags.append(language)
        trace_id = system.episodic.log_trace(
            content=f"[Code Pattern: {name}] {content[:500]}",
            kind="code_pattern",
            tags=ep_tags,
            salience=0.8,
        )

        return json.dumps(
            {
                "action": "store",
                "name": name,
                "trace_id": trace_id,
                "language": language,
                "framework": framework,
                "category": category,
                "confidence": meta.confidence,
                "message": f"Pattern '{name}' stored with structured metadata.",
            }
        )

    elif action == "get":
        skill_content = system.procedural.get_skill(name)
        if skill_content:
            meta = system.procedural.get_skill_meta(name)
            result: dict = {
                "action": "get",
                "name": name,
                "content": skill_content,
            }
            if meta:
                result["metadata"] = meta.to_dict()
            return json.dumps(result)
        else:
            return json.dumps(
                {
                    "action": "get",
                    "name": name,
                    "content": None,
                    "message": f"Pattern '{name}' not found.",
                }
            )

    else:
        return json.dumps(
            {"error": f"Unknown action '{action}'. Use 'store' or 'get'."}
        )


@mcp.tool()
@_safe_json
def engram_debug_log(
    error_message: str,
    resolution: str,
    stack_trace: str = "",
    error_category: str = "logic",
    resolution_strategy: str = "",
    attempt_history: str = "",
    associated_adr: str = "",
) -> str:
    """Record a debugging episode with error and resolution.

    Captures the error context and successful fix, enabling the agent
    to recall resolution strategies for similar future failures.
    Error fingerprinting groups semantically similar issues.

    Implements Table 3 (Episodic Debugging Memory schema) from the
    buildplan: fingerprint_id, error_category, attempt_history,
    resolution_strategy, associated_adr.

    Args:
        error_message: The error message or description of the problem.
        resolution: How the error was resolved (description + key changes).
        stack_trace: The stack trace (optional, for richer context).
        error_category: One of: logic, integration, performance, security, configuration.
        resolution_strategy: High-level summary of the fix approach
            (e.g., "Implement lazy loading", "Add retry with backoff").
            Distinct from the detailed resolution — this is the reusable
            *pattern* that solved the problem.
        attempt_history: JSON string or prose describing failed approaches
            before the successful fix.  Records what was tried and why
            it was rejected, so the agent avoids repeating dead ends.
        associated_adr: Trace ID of a related Architecture Decision Record.
            Links the debug session to the design decision that caused or
            resolved the issue.
    """
    from engram.extraction.fingerprint import analyze_error

    system = _get_system()

    valid_categories = {
        "logic",
        "integration",
        "performance",
        "security",
        "configuration",
    }
    if error_category not in valid_categories:
        error_category = "logic"

    # --- B1: Sentry-style error fingerprinting ---
    analysis = analyze_error(error_message, stack_trace, error_category)
    fingerprint = analysis["fingerprint"]

    # Search for prior debug sessions with the same fingerprint
    prior_sessions = _find_by_fingerprint(system, fingerprint)

    # --- Build structured debug content ---
    debug_content = (
        f"## Debug Session: {error_category}\n\n### Error\n{error_message}\n\n"
    )
    if stack_trace:
        debug_content += f"### Stack Trace\n```\n{stack_trace}\n```\n\n"
    if attempt_history:
        debug_content += f"### Attempt History\n{attempt_history}\n\n"
    debug_content += f"### Resolution\n{resolution}\n"
    if resolution_strategy:
        debug_content += f"\n### Resolution Strategy\n{resolution_strategy}\n"

    # --- Build metadata (Table 3 schema) ---
    trace_metadata = {
        "fingerprint": fingerprint,
        "exception_type": analysis["exception_type"],
        "message_template": analysis["message_template"],
        "app_frames": analysis["app_frames"],
    }
    if resolution_strategy:
        trace_metadata["resolution_strategy"] = resolution_strategy
    if attempt_history:
        # Store attempt_history as-is — could be JSON or prose
        trace_metadata["attempt_history"] = attempt_history
    if associated_adr:
        trace_metadata["associated_adr"] = associated_adr

    trace_id = system.episodic.log_trace(
        content=debug_content,
        kind="debug_session",
        tags=["debug", error_category],
        salience=0.75,
        **trace_metadata,
    )

    result = {
        "trace_id": trace_id,
        "kind": "debug_session",
        "error_category": error_category,
        "fingerprint": fingerprint,
        "exception_type": analysis["exception_type"],
        "message_template": analysis["message_template"],
        "app_frames": analysis["app_frames"],
        "frame_count": analysis["frame_count"],
        "message": "Debug session logged — will be recalled for similar errors.",
    }
    if resolution_strategy:
        result["resolution_strategy"] = resolution_strategy
    if associated_adr:
        result["associated_adr"] = associated_adr
    if prior_sessions:
        result["prior_matches"] = len(prior_sessions)
        result["prior_trace_ids"] = [s["id"] for s in prior_sessions[:5]]
        result["message"] = (
            f"Debug session logged. Found {len(prior_sessions)} prior "
            f"session(s) with the same error fingerprint — check prior "
            f"resolutions before reinventing the fix."
        )
    return json.dumps(result)


@mcp.tool()
@_safe_json
def engram_debug_recall(
    fingerprint: str = "",
    error_message: str = "",
    error_category: str = "",
    limit: int = 10,
) -> str:
    """Recall prior debug sessions by fingerprint or error pattern.

    Use this BEFORE attempting a fix to check if the same (or similar)
    error has been resolved before.  Returns matching debug sessions
    with their resolutions, strategies, and attempt histories.

    Lookup modes (use one):
      - fingerprint: Exact fingerprint match (fastest, most precise).
      - error_message: Computes the fingerprint from the message and
        searches by that.  Also falls back to FTS text search.
      - error_category: Browse all debug sessions in a category.

    Args:
        fingerprint: Exact SHA-256 fingerprint to look up.
        error_message: Error message to fingerprint and search for.
        error_category: Filter by category (logic, integration, etc.).
        limit: Maximum results to return (default 10).
    """
    system = _get_system()
    results = []

    # Mode 1: Exact fingerprint lookup
    if fingerprint:
        results = _find_by_fingerprint(system, fingerprint, limit)

    # Mode 2: Compute fingerprint from error message, then lookup + FTS fallback
    elif error_message:
        from engram.extraction.fingerprint import compute_fingerprint as _fp

        # Try fingerprint with category if provided, then all categories
        # (since we don't know which category was used at log time).
        categories_to_try = []
        if error_category:
            categories_to_try.append(error_category)
        categories_to_try.extend(
            [
                "logic",
                "integration",
                "performance",
                "security",
                "configuration",
            ]
        )
        # Deduplicate while preserving order
        seen = set()
        for cat in categories_to_try:
            if cat in seen:
                continue
            seen.add(cat)
            computed_fp = _fp(error_message, "", cat)
            results = _find_by_fingerprint(system, computed_fp, limit)
            if results:
                break

        # Also try without category (bare fingerprint)
        if not results:
            computed_fp = _fp(error_message)
            results = _find_by_fingerprint(system, computed_fp, limit)

        # FTS fallback if fingerprint match found nothing
        if not results:
            try:
                results = system.episodic.search_traces(error_message, limit=limit)
                # Filter to debug_session kind only
                results = [r for r in results if r.get("kind") == "debug_session"]
            except Exception:
                pass

    # Mode 3: Browse by category
    elif error_category:
        try:
            rows = system.episodic.conn.execute(
                """
                SELECT * FROM traces
                WHERE kind = 'debug_session'
                  AND json_extract(tags, '$') LIKE ?
                ORDER BY created DESC
                LIMIT ?
                """,
                (f'%"{error_category}"%', limit),
            ).fetchall()
            results = [system.episodic._row_to_dict(r) for r in rows]
        except Exception:
            pass

    if not results:
        return json.dumps(
            {
                "matches": 0,
                "sessions": [],
                "message": "No matching debug sessions found.",
            }
        )

    # Format results — extract key fields from metadata
    sessions = []
    for r in results[:limit]:
        meta = {}
        if r.get("metadata"):
            try:
                meta = (
                    json.loads(r["metadata"])
                    if isinstance(r["metadata"], str)
                    else r["metadata"]
                )
            except (json.JSONDecodeError, TypeError):
                pass

        session = {
            "trace_id": r.get("id", ""),
            "created": r.get("created", ""),
            "error_category": meta.get("error_category", ""),
            "fingerprint": meta.get("fingerprint", ""),
            "exception_type": meta.get("exception_type", ""),
            "resolution_strategy": meta.get("resolution_strategy", ""),
            "attempt_history": meta.get("attempt_history", ""),
            "associated_adr": meta.get("associated_adr", ""),
            "content_preview": (r.get("content", ""))[:500],
        }
        sessions.append(session)

    return json.dumps(
        {
            "matches": len(sessions),
            "sessions": sessions,
            "message": f"Found {len(sessions)} matching debug session(s).",
        }
    )


def _find_by_fingerprint(
    system: "MemorySystem",
    fingerprint: str,
    limit: int = 10,
) -> list:
    """Find prior debug_session traces with a matching fingerprint."""
    try:
        rows = system.episodic.conn.execute(
            """
            SELECT * FROM traces
            WHERE kind = 'debug_session'
              AND json_extract(metadata, '$.fingerprint') = ?
            ORDER BY created DESC
            LIMIT ?
            """,
            (fingerprint, limit),
        ).fetchall()
        return [system.episodic._row_to_dict(r) for r in rows]
    except Exception:
        # Metadata column may not exist in older schemas, degrade gracefully
        return []


@mcp.tool()
@_safe_json
def engram_get_rules(file_path: str = "") -> str:
    """Retrieve coding standards and style rules for the current project.

    Returns the project's coding philosophy (from SOUL.md), forbidden
    constructs, review checklist, and any relevant procedural patterns.
    Optionally scoped to a specific file path for directory-aware rules.

    Args:
        file_path: Optional file path to scope rules to (for future
                   per-directory rule support).
    """
    system = _get_system()

    # Load SOUL.md (coding philosophy / standards)
    soul_text = system.semantic.get_identity()

    # Load preferences (which include coding style preferences)
    prefs = system.semantic.get_preferences()

    # Load any style-related procedural skills
    style_skills = system.procedural.search_skills("coding style convention standard")

    result = {
        "coding_philosophy": soul_text[:2000] if soul_text else "",
        "preferences": prefs[:500] if prefs else "",
        "style_skills": style_skills[:5] if style_skills else [],
    }

    if file_path:
        result["scoped_to"] = file_path

    return json.dumps(result, indent=2)


@mcp.tool()
@_safe_json
def engram_add_wiring(
    subject: str,
    predicate: str,
    object: str,
    confidence: float = 0.9,
) -> str:
    """Record a code dependency relationship in the knowledge graph.

    Builds the project's wiring map by recording structural relationships
    between code entities.  Enables multi-hop reasoning: e.g., "what
    modules depend on the database schema?"

    Supported predicates:
      - depends_on:  A depends on B (import, API call, etc.)
      - exports:     A exports B (function, class, constant)
      - validates:   A validates B (test file -> source file)
      - supersedes:  A supersedes B (ADR versioning, API migration)
      - uses:        A uses B (lighter than depends_on)
      - implements:  A implements B (interface realization)

    Args:
        subject: Source entity (e.g., "auth.py", "UserService", "v2.0").
        predicate: Relationship type (see above).
        object: Target entity (e.g., "database.py", "verify_token").
        confidence: Confidence in this relationship (0-1, default 0.9).
    """
    system = _get_system()

    valid_predicates = {
        "depends_on",
        "exports",
        "validates",
        "supersedes",
        "uses",
        "implements",
    }
    if predicate not in valid_predicates:
        return json.dumps(
            {
                "error": f"Unknown predicate '{predicate}'. "
                f"Valid predicates: {sorted(valid_predicates)}",
            }
        )

    rel_id = system.episodic.add_relationship(
        subject=subject,
        predicate=predicate,
        object=object,
        confidence=_clamp(confidence),
    )

    return json.dumps(
        {
            "relationship_id": rel_id,
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "confidence": confidence,
            "message": f"Wiring recorded: {subject} --[{predicate}]--> {object}",
        }
    )


@mcp.tool()
@_safe_json
def engram_project_init(
    project_name: str,
    description: str = "",
    languages: str = "",
    patterns: str = "",
) -> str:
    """Initialize or switch to a project scope for memory isolation.

    Sets the active project context so subsequent memory operations
    (traces, messages, sessions) are scoped to this project.  Also
    creates a project_context trace capturing the project metadata.

    Call this when starting work on a specific project or repository.

    Args:
        project_name: Project identifier (e.g., "engram", "my-api").
        description: Brief description of the project.
        languages: Comma-separated languages (e.g., "python,typescript").
        patterns: Comma-separated architectural patterns used.
    """
    global _current_project
    system = _get_system()

    # Track project in both session registry and legacy global
    session = _get_session()
    session.project = project_name
    _current_project = project_name

    # Store project context as a high-salience trace
    context_parts = [f"# Project: {project_name}"]
    if description:
        context_parts.append(f"\n{description}")
    if languages:
        context_parts.append(f"\nLanguages: {languages}")
    if patterns:
        context_parts.append(f"\nPatterns: {patterns}")

    content = "\n".join(context_parts)
    tags = ["project", project_name]
    if languages:
        tags.extend(l.strip() for l in languages.split(",") if l.strip())

    trace_id = system.episodic.log_trace(
        content=content,
        kind="project_context",
        tags=tags,
        salience=0.9,
        project=project_name,
    )

    # Load existing ADRs for this project
    adrs = system.episodic.get_traces_by_kind(
        "architecture_decision",
        limit=10,
        project=project_name,
    )

    return json.dumps(
        {
            "project": project_name,
            "trace_id": trace_id,
            "active_adrs": len(adrs),
            "message": f"Project '{project_name}' initialized. "
            f"Memory operations now scoped to this project.",
        }
    )


# ===========================================================================
# AST / Code Analysis Tools
# ===========================================================================


@mcp.tool()
@_safe_json
def engram_extract_symbols(
    code: str,
    language: str = "",
    file_path: str = "",
    store: bool = True,
) -> str:
    """Extract code symbols using AST analysis and optionally store them.

    Runs tree-sitter (JS/TS) or stdlib ast (Python) parsing to extract
    functions, classes, imports, complexity metrics, and anti-patterns
    from source code.  Optionally stores the extracted symbols as a
    trace for future retrieval.

    This is the on-demand version of what happens automatically in the
    after-pipeline (which extracts symbols from LLM response code blocks).

    Args:
        code: Source code to analyze.
        language: Language hint (auto-detected if empty).
        file_path: Optional file path for context.
        store: If True, store extracted symbols as a code_symbols trace.
    """
    try:
        from engram.extraction.ast_engine import analyze_code
    except ImportError:
        return json.dumps(
            {"error": "AST extraction not available. Install tree-sitter."}
        )

    analysis = analyze_code(code, language)

    result = analysis.to_dict()
    result["repo_map"] = analysis.to_repo_map()
    if file_path:
        result["file_path"] = file_path

    # Optionally store as a trace
    if store and analysis.symbols:
        system = _get_system()
        content_parts = []
        if file_path:
            content_parts.append(f"# Symbols: {file_path}")
        content_parts.append(analysis.to_repo_map())
        if analysis.imports:
            content_parts.append(
                "\nImports: "
                + ", ".join(i.module for i in analysis.imports if i.module)
            )

        tags = ["ast", "symbols"]
        if file_path:
            tags.append(file_path.split("/")[-1].split("\\")[-1])
        if analysis.language != "unknown":
            tags.append(analysis.language)

        trace_id = system.episodic.log_trace(
            content="\n".join(content_parts),
            kind="code_symbols",
            tags=tags,
            salience=0.6,
            project=_get_session().project,
            file_path=file_path,
            language=analysis.language,
            fingerprint=analysis.fingerprint,
            symbol_count=len(analysis.symbols),
            import_count=len(analysis.imports),
        )
        result["trace_id"] = trace_id

        # Auto-store wiring edges from imports
        for imp in analysis.imports:
            if imp.module and file_path:
                try:
                    system.episodic.add_relationship(
                        subject=file_path,
                        predicate="depends_on",
                        object=imp.module,
                        confidence=0.85,
                        source_trace_id=trace_id,
                    )
                except Exception:
                    pass  # Non-critical — don't fail on wiring

        # Auto-store export edges
        for sym in analysis.exported_symbols:
            if file_path and sym.kind.value in ("function", "class", "interface"):
                try:
                    system.episodic.add_relationship(
                        subject=file_path,
                        predicate="exports",
                        object=sym.qualified_name,
                        confidence=0.95,
                        source_trace_id=trace_id,
                    )
                except Exception:
                    pass

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
@_safe_json
def engram_repo_map(
    directory: str = "",
    max_tokens: int = 2000,
    extensions: str = "",
) -> str:
    """Generate an Aider-style repo-map for a directory.

    Indexes all supported source files in the directory using AST
    extraction and produces a compact representation of the repository
    structure showing file paths, class definitions, and function
    signatures.  Ideal for providing project context to LLMs.

    The map is automatically stored as a project_context trace for
    future retrieval.

    Args:
        directory: Directory to scan. Defaults to current project root.
        max_tokens: Approximate token budget for the map (default 2000).
        extensions: Comma-separated file extensions (e.g., ".py,.ts").
                    Defaults to common code extensions.
    """
    try:
        from engram.extraction.symbol_index import SymbolIndex
    except ImportError:
        return json.dumps({"error": "Symbol index not available."})

    import os

    if not directory:
        directory = os.getcwd()

    if not os.path.isdir(directory):
        return json.dumps({"error": f"Directory not found: {directory}"})

    # Parse extensions
    ext_set = None
    if extensions:
        ext_set = {
            e.strip() if e.strip().startswith(".") else f".{e.strip()}"
            for e in extensions.split(",")
            if e.strip()
        }

    index = SymbolIndex()
    file_count = index.index_directory(directory, extensions=ext_set)

    repo_map = index.generate_repo_map(max_tokens=max_tokens)

    # Store as project context trace
    system = _get_system()
    current_project = _get_session().project
    project_name = current_project or os.path.basename(directory)

    trace_id = system.episodic.log_trace(
        content=f"# Repo Map: {project_name}\n\n{repo_map}",
        kind="project_context",
        tags=["repo_map", "ast", project_name],
        salience=0.8,
        project=current_project,
        directory=directory,
        file_count=file_count,
        symbol_count=index.symbol_count,
    )

    # Store dependency edges
    edges = index.get_dependency_edges()
    stored_edges = 0
    for edge in edges[:200]:  # Cap at 200 edges to avoid flooding
        try:
            system.episodic.add_relationship(
                subject=edge["source"],
                predicate=edge["predicate"],
                object=edge["target"],
                confidence=0.85,
                source_trace_id=trace_id,
            )
            stored_edges += 1
        except Exception:
            pass

    return json.dumps(
        {
            "directory": directory,
            "file_count": file_count,
            "symbol_count": index.symbol_count,
            "trace_id": trace_id,
            "edges_stored": stored_edges,
            "repo_map": repo_map,
            "message": (
                f"Indexed {file_count} files with {index.symbol_count} symbols. "
                f"Stored {stored_edges} dependency edges."
            ),
        },
        indent=2,
    )


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
# Multi-Agent Session Tools
# ---------------------------------------------------------------------------


@mcp.tool()
@_safe_json
def engram_sessions() -> str:
    """List all active client sessions.

    Returns diagnostic information about connected MCP clients,
    including session IDs, client names, active person/project,
    and session age.  Useful for debugging multi-agent setups.
    """
    sessions = _sessions.list_sessions()
    return json.dumps(
        {"active_sessions": sessions, "count": len(sessions)},
        indent=2,
        default=str,
    )


@mcp.tool()
@_safe_json
def engram_register_client(
    client_name: str = "",
    client_version: str = "",
    session_id: str = "",
) -> str:
    """Register or identify an MCP client session.

    Call this at the start of a multi-agent session to associate
    a human-readable client name with the current session.  Not
    required for single-client (stdio) mode.

    Args:
        client_name: Human-readable client name (e.g., "cursor", "claude-code").
        client_version: Client version string.
        session_id: Explicit session ID (auto-generated if empty).
    """
    import uuid

    sid = session_id or str(uuid.uuid4())[:8]
    set_current_session_id(sid)
    session = _get_session()
    session.client_name = client_name
    session.client_version = client_version
    log.info(
        "Registered client: %s v%s (session=%s)",
        client_name,
        client_version,
        sid,
    )
    return json.dumps(session.to_dict(), indent=2, default=str)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _shutdown() -> None:
    """Clean up the global MemorySystem and session registry on exit."""
    global _system
    if _system is not None:
        log.info("Shutting down Engram memory system...")
        active = _sessions.count()
        if active > 1:
            log.info("Closing %d active client sessions", active)
        try:
            _system.close()
        except Exception as exc:
            log.warning("Error during shutdown: %s", exc)
        _system = None


def run_server(
    data_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    transport: str = "stdio",
    host: str = "0.0.0.0",
    port: int = 8765,
) -> None:
    """Initialize and run the MCP server.

    Transports:
        stdio (default):
            Single-client mode.  Each IDE spawns its own Engram
            process.  Client sessions are isolated at the OS level.
        streamable-http:
            Multi-client HTTP mode.  Multiple IDEs connect to a
            single Engram server.  Each client gets its own session
            via ``engram_register_client()`` or automatic session
            management by the MCP transport layer.
        sse:
            Legacy Server-Sent Events mode.  Similar to streamable-http
            but uses SSE for the server→client channel.

    Args:
        data_dir: Path to the Engram data directory.
        config_path: Path to YAML config file.
        transport: MCP transport: "stdio", "streamable-http", or "sse".
        host: Bind address for HTTP transports (default: 0.0.0.0).
        port: Port for HTTP transports (default: 8765).
    """
    system = init_system(data_dir=data_dir, config_path=config_path)
    atexit.register(_shutdown)
    log.info(
        "Starting Engram MCP server (transport=%s, sessions=multi-agent)",
        transport,
    )
    if transport in ("streamable-http", "sse"):
        # Configure FastMCP's host/port settings for HTTP transports.
        # These are read by run_sse_async() and run_streamable_http_async()
        # when creating the uvicorn server.
        mcp.settings.host = host
        mcp.settings.port = port
        log.info("HTTP endpoint: http://%s:%d", host, port)
    try:
        mcp.run(transport=transport)  # type: ignore[arg-type]
    finally:
        _shutdown()
