"""
engram.core.types — Data types for the engram memory system.

Every structure here is a plain dataclass: no ORM, no magic,
serialisable to dict/JSON in one call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional
import uuid

from engram.core.tokens import estimate_tokens  # single source of truth

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Canonical type for LLM callables throughout engram.
#: Signature: ``(prompt: str, system: str) -> str``
LLMFunc = Callable[[str, str], str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_id() -> str:
    """12-hex-char unique identifier."""
    return uuid.uuid4().hex[:12]


def now_iso() -> str:
    """Current UTC timestamp in ISO-8601 with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Trace — atomic unit of episodic memory
# ---------------------------------------------------------------------------

TRACE_KINDS = frozenset(
    {
        # --- Core memory kinds (retained from v1) ---
        "episode",  # Generic episodic memory
        "realization",  # Architectural insight or important discovery
        "factual",  # Learned fact or piece of knowledge
        "reflection",  # Metacognitive observation
        # --- Consolidation hierarchy (MemGPT-inspired) ---
        "summary",  # Compacted conversation summary
        "thread",  # Multi-episode thematic thread
        "arc",  # Long-term project/growth arc
        # --- Infrastructure ---
        "temporal",  # Time-decaying memory with revival mechanics
        "utility",  # RL Q-value scored memory (learns its own usefulness)
        "workspace_eviction",  # Item evicted from cognitive workspace
        # --- Code-first kinds (v2 pivot) ---
        "code_pattern",  # Validated reusable implementation pattern
        "debug_session",  # Error -> investigation -> fix episode
        "architecture_decision",  # ADR: decision + rationale + alternatives
        "wiring_map",  # Cross-file dependency snapshot
        "error_resolution",  # Specific error + resolution pair
        "test_strategy",  # Testing approach for a module/feature
        "project_context",  # Project-level metadata snapshot
        "code_review",  # Observations from code review
        "code_symbols",  # AST-extracted symbol index for a file/module
    }
)


@dataclass
class Trace:
    """
    Atomic unit of episodic memory.

    A trace records a single moment of experience — something said,
    felt, realised, or corrected.  Traces are the rows of the episodic
    store and the primary unit of salience decay / reinforcement.
    """

    content: str
    kind: str = "episode"
    tags: List[str] = field(default_factory=list)
    salience: float = 0.5
    metadata: Dict = field(default_factory=dict)
    project: str = ""  # Project scope (empty = global)

    # auto-populated
    id: str = field(default_factory=generate_id)
    created: str = field(default_factory=now_iso)
    tokens: int = 0
    access_count: int = 0
    last_accessed: str = field(default_factory=now_iso)

    def __post_init__(self) -> None:
        if self.kind not in TRACE_KINDS:
            raise ValueError(
                f"Invalid trace kind {self.kind!r}; "
                f"expected one of {sorted(TRACE_KINDS)}"
            )
        if self.tokens == 0:
            self.tokens = estimate_tokens(self.content)
        self.salience = max(0.0, min(1.0, self.salience))

    # -- access tracking ----------------------------------------------------

    def touch(self) -> None:
        """Record an access (context-load) of this trace."""
        self.access_count += 1
        self.last_accessed = now_iso()

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "created": self.created,
            "kind": self.kind,
            "tags": list(self.tags),
            "salience": round(self.salience, 4),
            "tokens": self.tokens,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Trace":
        return cls(
            id=d["id"],
            content=d["content"],
            created=d.get("created", now_iso()),
            kind=d.get("kind", "episode"),
            tags=d.get("tags", []),
            salience=d.get("salience", 0.5),
            tokens=d.get("tokens", 0),
            access_count=d.get("access_count", 0),
            last_accessed=d.get("last_accessed", now_iso()),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Message — a single conversational turn
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """
    One turn in a conversation.

    `person` is the canonical name of the other party.
    `speaker` is who actually said this — either the person's name or
    ``"self"`` when *we* spoke.
    """

    person: str
    speaker: str
    content: str
    source: str = "direct"
    salience: float = 0.5
    signal: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)

    # auto-populated
    id: str = field(default_factory=generate_id)
    timestamp: str = field(default_factory=now_iso)

    def __post_init__(self) -> None:
        self.salience = max(0.0, min(1.0, self.salience))

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "person": self.person,
            "speaker": self.speaker,
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp,
            "salience": round(self.salience, 4),
            "signal": self.signal,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Message":
        return cls(
            id=d["id"],
            person=d["person"],
            speaker=d["speaker"],
            content=d["content"],
            source=d.get("source", "direct"),
            timestamp=d.get("timestamp", now_iso()),
            salience=d.get("salience", 0.5),
            signal=d.get("signal"),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Signal — Code Quality Signal (CQS)
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """
    Four-facet Code Quality Signal (CQS).

    Each facet is 0-1:
      correctness  — syntactic validity, type correctness, logic soundness
      consistency  — adherence to project patterns, naming, import style
      completeness — error handling, edge cases, tests, documentation
      robustness   — input validation, resource cleanup, error boundaries

    Derived properties give a quick read on overall health.

    NOTE: Field names changed in v2 pivot (was alignment/embodiment/
    clarity/vitality).  The dataclass structure, Signal type, and
    SignalTracker are load-bearing infrastructure used throughout the
    pipeline — only the facet semantics and measurement changed.
    """

    correctness: float = 0.5
    consistency: float = 0.5
    completeness: float = 0.5
    robustness: float = 0.5
    trace_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for attr in ("correctness", "consistency", "completeness", "robustness"):
            setattr(self, attr, max(0.0, min(1.0, getattr(self, attr))))

    # -- derived properties -------------------------------------------------

    # CQS weights: correctness matters most (does it work?), then
    # consistency (does it fit?), completeness and robustness tied.
    _WEIGHTS = {
        "correctness": 0.40,
        "consistency": 0.20,
        "completeness": 0.20,
        "robustness": 0.20,
    }

    @property
    def health(self) -> float:
        """Weighted CQS score (correctness 40%, consistency 20%, completeness 20%, robustness 20%)."""
        return (
            self.correctness * self._WEIGHTS["correctness"]
            + self.consistency * self._WEIGHTS["consistency"]
            + self.completeness * self._WEIGHTS["completeness"]
            + self.robustness * self._WEIGHTS["robustness"]
        )

    @property
    def state(self) -> str:
        """Human-readable quality state label."""
        h = self.health
        if h >= 0.75:
            return "excellent"
        if h >= 0.50:
            return "acceptable"
        if h >= 0.35:
            return "needs_improvement"
        return "poor"

    @property
    def needs_correction(self) -> bool:
        """True when health drops below 0.5."""
        return self.health < 0.5

    @property
    def _facets(self) -> Dict[str, float]:
        return {
            "correctness": self.correctness,
            "consistency": self.consistency,
            "completeness": self.completeness,
            "robustness": self.robustness,
        }

    @property
    def weakest_facet(self) -> str:
        """Name of the lowest-scoring facet."""
        return min(self._facets, key=self._facets.get)  # type: ignore[arg-type]

    @property
    def strongest_facet(self) -> str:
        """Name of the highest-scoring facet."""
        return max(self._facets, key=self._facets.get)  # type: ignore[arg-type]

    @property
    def polarity_gap(self) -> float:
        """Difference between strongest and weakest facet."""
        vals = self._facets.values()
        return max(vals) - min(vals)

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "correctness": round(self.correctness, 4),
            "consistency": round(self.consistency, 4),
            "completeness": round(self.completeness, 4),
            "robustness": round(self.robustness, 4),
            "health": round(self.health, 4),
            "state": self.state,
            "needs_correction": self.needs_correction,
            "weakest_facet": self.weakest_facet,
            "polarity_gap": round(self.polarity_gap, 4),
            "trace_ids": list(self.trace_ids),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Signal":
        return cls(
            correctness=d.get("correctness", d.get("alignment", 0.5)),
            consistency=d.get("consistency", d.get("embodiment", 0.5)),
            completeness=d.get("completeness", d.get("clarity", 0.5)),
            robustness=d.get("robustness", d.get("vitality", 0.5)),
            trace_ids=d.get("trace_ids", []),
        )


# ---------------------------------------------------------------------------
# Context — result of the "before" pipeline
# ---------------------------------------------------------------------------


@dataclass
class Context:
    """
    Assembled context ready for injection into the system prompt.

    Produced by the *before-pipeline*: identity load, memory recall,
    relationship lookup, salience ranking, budget trimming.

    ``timings`` holds step-level latency in seconds (OB-1).
    """

    text: str
    trace_ids: List[str] = field(default_factory=list)
    person: str = ""
    tokens_used: int = 0
    token_budget: int = 6000
    memories_loaded: int = 0
    timings: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tokens_used == 0 and self.text:
            self.tokens_used = estimate_tokens(self.text)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.token_budget - self.tokens_used)

    @property
    def budget_utilisation(self) -> float:
        if self.token_budget == 0:
            return 0.0
        return self.tokens_used / self.token_budget

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "trace_ids": list(self.trace_ids),
            "person": self.person,
            "tokens_used": self.tokens_used,
            "token_budget": self.token_budget,
            "memories_loaded": self.memories_loaded,
            "budget_remaining": self.budget_remaining,
            "timings": {k: round(v, 4) for k, v in self.timings.items()},
        }


# ---------------------------------------------------------------------------
# AfterResult — result of the "after" pipeline
# ---------------------------------------------------------------------------


@dataclass
class AfterResult:
    """
    What the *after-pipeline* produces: a code quality signal reading,
    salience estimate, semantic updates, and the IDs of the logged
    message and (optional) trace.

    ``timings`` holds step-level latency in seconds (OB-1).
    """

    signal: Signal
    salience: float = 0.5
    updates: List[Dict] = field(default_factory=list)
    logged_message_id: str = ""
    logged_trace_id: Optional[str] = None
    timings: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.salience = max(0.0, min(1.0, self.salience))

    def to_dict(self) -> Dict:
        return {
            "signal": self.signal.to_dict(),
            "salience": round(self.salience, 4),
            "updates": list(self.updates),
            "logged_message_id": self.logged_message_id,
            "logged_trace_id": self.logged_trace_id,
            "timings": {k: round(v, 4) for k, v in self.timings.items()},
        }


# ---------------------------------------------------------------------------
# MemoryStats — system health snapshot
# ---------------------------------------------------------------------------


@dataclass
class MemoryStats:
    """
    High-level health metrics for the memory system.

    `memory_pressure` is 0-1 indicating how close the episodic store is
    to its configured max_traces limit.
    """

    episodic_count: int = 0
    semantic_facts: int = 0
    procedural_skills: int = 0
    total_messages: int = 0
    avg_salience: float = 0.0
    memory_pressure: float = 0.0
    status: str = ""

    def __post_init__(self) -> None:
        self.memory_pressure = max(0.0, min(1.0, self.memory_pressure))
        if not self.status:
            self.status = self._compute_status()

    def _compute_status(self) -> str:
        if self.memory_pressure > 0.9:
            return "critical"
        if self.memory_pressure > 0.7:
            return "warning"
        return "ok"

    @property
    def total_memories(self) -> int:
        return self.episodic_count + self.semantic_facts + self.procedural_skills

    def to_dict(self) -> Dict:
        return {
            "episodic_count": self.episodic_count,
            "semantic_facts": self.semantic_facts,
            "procedural_skills": self.procedural_skills,
            "total_messages": self.total_messages,
            "total_memories": self.total_memories,
            "avg_salience": round(self.avg_salience, 4),
            "memory_pressure": round(self.memory_pressure, 4),
            "status": self.status,
        }
