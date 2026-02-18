"""
engram.core.config — Configuration for the engram memory system.

Supports loading from YAML, environment variables, and programmatic
construction.  LLM provider functions are built lazily so optional
dependencies (httpx, openai, anthropic) are only imported when needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from engram.core.types import LLMFunc


# ---------------------------------------------------------------------------
# LLM usage tracking (OB-2)
# ---------------------------------------------------------------------------


class LLMUsageTracker:
    """Accumulates LLM token usage across calls.

    Thread-safe via simple attribute accumulation (GIL-protected).
    Call ``snapshot()`` to get current totals without resetting.
    """

    __slots__ = ("total_calls", "total_input_tokens", "total_output_tokens")

    def __init__(self) -> None:
        self.total_calls: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def snapshot(self) -> Dict[str, int]:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }


# Singleton tracker — importable by other modules for inspection
llm_usage = LLMUsageTracker()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """
    Central configuration object.

    Construct directly, via ``Config.from_yaml(path)``, or via
    ``Config.from_data_dir(path)`` for quick bootstrap.
    """

    # -- storage ------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path("./engram_data"))

    # -- owner --------------------------------------------------------------
    core_person: str = ""  # canonical name of the owner (backward-compat alias)
    default_project: str = ""  # default project name for scoping

    # -- signal / extraction modes ------------------------------------------
    signal_mode: str = "hybrid"  # "hybrid" | "regex" | "llm"
    extract_mode: str = "off"  # "llm" | "off"

    # -- LLM provider -------------------------------------------------------
    llm_provider: str = "ollama"  # "ollama" | "openai" | "anthropic" | "custom"
    llm_model: str = "llama3.2"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("ENGRAM_LLM_API_KEY")
    )
    llm_func: Optional[LLMFunc] = field(default=None, repr=False)
    llm_weight: float = 0.6  # weight for blended (regex+llm) signal

    # -- embedding model ----------------------------------------------------
    # Model used for semantic (vector) search.  When set and the LLM
    # provider is "ollama", an Ollama-based embedding function is built
    # automatically.  When empty, ChromaDB falls back to its built-in
    # default (all-MiniLM-L6-v2, 384d, MTEB ~0.56).
    #
    # Default: "nomic-embed-text" — 768d, MTEB ~0.63, free, runs locally.
    #          Requires: ollama pull nomic-embed-text
    # Alternative: "mxbai-embed-large" (1024d, MTEB ~0.64, larger).
    # Set to "" to fall back to ChromaDB's built-in default.
    embedding_model: str = "nomic-embed-text"

    # -- cross-encoder reranking -------------------------------------------
    # When enabled, search results are rescored by a cross-encoder after
    # RRF merge.  This is the #1 RAG optimization (~15-30% precision gain)
    # but adds ~30ms latency per search.
    #
    # Requires: pip install sentence-transformers
    # Model runs on CPU by default; set reranker_device="cuda" for GPU.
    reranker_enabled: bool = True  # auto-disabled if deps missing
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    reranker_device: str = ""  # "" = auto (CPU for small models)

    # -- memory budget & decay ----------------------------------------------
    token_budget: int = 12000  # v2: increased for code context density
    decay_half_life_hours: float = 168.0  # 1 week
    max_traces: int = 50_000
    reinforce_delta: float = 0.05
    weaken_delta: float = 0.03

    # -- context budget shares (must sum to 1.0) ----------------------------
    identity_share: float = 0.16
    relationship_share: float = 0.12
    grounding_share: float = 0.10
    recent_conversation_share: float = 0.22
    episodic_share: float = 0.16
    procedural_share: float = 0.06
    reserve_share: float = 0.18

    # -- signal thresholds --------------------------------------------------
    signal_health_threshold: float = 0.45  # below this, grounding context warns
    reinforce_threshold: float = 0.7
    weaken_threshold: float = 0.4

    # -- compaction settings (MemGPT-inspired) ------------------------------
    compaction_keep_recent: int = 20  # messages per person to leave untouched
    compaction_segment_size: int = 30  # max messages per summary segment
    compaction_min_messages: int = 40  # threshold before compaction triggers

    # -- hierarchical consolidation settings --------------------------------
    consolidation_min_episodes: int = 5  # min episodes before thread creation
    consolidation_time_window_hours: float = 72.0  # temporal proximity for threads
    consolidation_min_threads: int = 3  # min threads before arc creation
    consolidation_max_episodes_per_run: int = 200  # cap per consolidation pass

    # -- cognitive workspace (Miller's Law 7±2) -----------------------------
    workspace_capacity: int = 7
    workspace_decay_rate: float = 0.95
    workspace_rehearsal_boost: float = 0.15
    workspace_expiry_threshold: float = 0.1

    # -- code-first boot sequence -------------------------------------------
    boot_n_sessions: int = 3  # recent coding session summaries at boot
    boot_n_decisions: int = 5  # recent ADRs at boot

    # -- structured logging (OB-3) -----------------------------------------
    structured_logging: bool = False  # emit JSON log lines when True

    # -- code quality signal ------------------------------------------------
    code_signal_mode: str = "regex"  # "regex" | "tool" | "hybrid"

    # -----------------------------------------------------------------------
    # Derived paths (all relative to data_dir)
    # -----------------------------------------------------------------------

    @property
    def soul_dir(self) -> Path:
        return self.data_dir / "soul"

    @property
    def semantic_dir(self) -> Path:
        return self.data_dir / "semantic"

    @property
    def procedural_dir(self) -> Path:
        return self.data_dir / "procedural"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "engram.db"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def soul_path(self) -> Path:
        return self.soul_dir / "SOUL.md"

    @property
    def identities_path(self) -> Path:
        return self.semantic_dir / "identities.yaml"

    @property
    def style_dir(self) -> Path:
        return self.data_dir / "style"

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def workspace_path(self) -> Path:
        return self.data_dir / "workspace.json"

    @property
    def consciousness_dir(self) -> Path:
        return self.data_dir / "consciousness"

    @property
    def runtime_dir(self) -> Path:
        return self.data_dir / "runtime"

    # -----------------------------------------------------------------------
    # Construction helpers
    # -----------------------------------------------------------------------

    def __post_init__(self) -> None:
        # normalise data_dir to an absolute Path
        self.data_dir = Path(self.data_dir).resolve()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file.

        Any key in the YAML that matches a Config field is applied.
        Unknown keys are silently ignored so the file can carry
        application-level settings alongside engram config.
        """
        import yaml  # optional dep — only needed for YAML config

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        # pull the engram section if nested, else use top-level
        data = raw.get("engram", raw)

        # convert data_dir string to Path
        if "data_dir" in data:
            data["data_dir"] = Path(data["data_dir"])

        # filter to known fields only
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}

        return cls(**filtered)

    @classmethod
    def from_data_dir(cls, data_dir: str | Path, **overrides: Any) -> "Config":
        """Quick constructor — just point at a data directory."""
        return cls(data_dir=Path(data_dir), **overrides)

    # -----------------------------------------------------------------------
    # Directory bootstrapping
    # -----------------------------------------------------------------------

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for d in (
            self.data_dir,
            self.soul_dir,
            self.semantic_dir,
            self.procedural_dir,
            self.embeddings_dir,
            self.style_dir,
            self.projects_dir,
            self.consciousness_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # LLM callable
    # -----------------------------------------------------------------------

    def get_llm_func(self) -> LLMFunc:
        """Return an ``(prompt, system) -> str`` callable for the configured provider.

        If ``llm_func`` was set directly (custom provider), it is
        returned as-is.  Otherwise a function is built from the
        provider / model / url / key settings.

        Dependencies (httpx, openai, anthropic) are imported lazily so
        they remain optional.
        """
        if self.llm_func is not None:
            return self.llm_func

        provider = self.llm_provider.lower()

        if provider == "ollama":
            return self._build_ollama_func()
        elif provider == "openai":
            return self._build_openai_func()
        elif provider == "anthropic":
            return self._build_anthropic_func()
        elif provider == "custom":
            raise ValueError(
                "llm_provider is 'custom' but no llm_func was provided. "
                "Pass a callable via Config(llm_func=my_func)."
            )
        else:
            raise ValueError(f"Unknown llm_provider: {provider!r}")

    # -- provider builders (private) ----------------------------------------

    def _build_ollama_func(self) -> LLMFunc:
        """Build LLM callable targeting Ollama's /api/generate."""
        try:
            import httpx  # noqa: F811
        except ImportError:
            raise ImportError(
                "httpx is required for the Ollama provider. "
                "Install it with:  pip install httpx"
            )

        base_url = self.llm_base_url.rstrip("/")
        model = self.llm_model
        # Create client once — reuse the connection pool across calls.
        client = httpx.Client(timeout=120.0)

        def ollama_call(prompt: str, system: str = "") -> str:
            payload: Dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "stream": False,
            }
            if system:
                payload["system"] = system

            resp = client.post(f"{base_url}/api/generate", json=payload)
            resp.raise_for_status()
            try:
                data = resp.json()
            except Exception:
                raise ValueError(
                    f"Ollama returned non-JSON response "
                    f"(status {resp.status_code}): {resp.text[:200]}"
                )
            # OB-2: Track token usage from Ollama response metadata
            llm_usage.record(
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
            )
            return data.get("response", "").strip()

        return ollama_call

    def _build_openai_func(self) -> LLMFunc:
        """Build LLM callable targeting an OpenAI-compatible API."""
        try:
            import openai  # noqa: F811
        except ImportError:
            raise ImportError(
                "openai is required for the OpenAI provider. "
                "Install it with:  pip install openai"
            )

        api_key = self.llm_api_key or os.environ.get("OPENAI_API_KEY", "")
        model = self.llm_model
        base_url = self.llm_base_url

        # Only set base_url on client if it differs from default Ollama URL,
        # meaning the user intentionally pointed at a custom endpoint.
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url and base_url != "http://localhost:11434":
            client_kwargs["base_url"] = base_url

        # Create client once — reuse the connection pool across calls.
        client = openai.OpenAI(**client_kwargs)

        def openai_call(prompt: str, system: str = "") -> str:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            if not response.choices:
                raise ValueError(
                    "OpenAI returned empty choices (content may have "
                    "been filtered). Check your prompt."
                )
            # OB-2: Track token usage from OpenAI response
            if response.usage:
                llm_usage.record(
                    input_tokens=response.usage.prompt_tokens or 0,
                    output_tokens=response.usage.completion_tokens or 0,
                )
            return (response.choices[0].message.content or "").strip()

        return openai_call

    def _build_anthropic_func(self) -> LLMFunc:
        """Build LLM callable targeting the Anthropic API."""
        try:
            import anthropic  # noqa: F811
        except ImportError:
            raise ImportError(
                "anthropic is required for the Anthropic provider. "
                "Install it with:  pip install anthropic"
            )

        api_key = self.llm_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        model = self.llm_model

        # Create client once — reuse the connection pool across calls.
        client = anthropic.Anthropic(api_key=api_key)

        def anthropic_call(prompt: str, system: str = "") -> str:
            kwargs: Dict[str, Any] = {
                "model": model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system

            response = client.messages.create(**kwargs)
            # OB-2: Track token usage from Anthropic response
            if hasattr(response, "usage") and response.usage:
                llm_usage.record(
                    input_tokens=getattr(response.usage, "input_tokens", 0),
                    output_tokens=getattr(response.usage, "output_tokens", 0),
                )
            # response.content is a list of content blocks
            parts = [block.text for block in response.content if hasattr(block, "text")]
            return "".join(parts).strip()

        return anthropic_call

    # -----------------------------------------------------------------------
    # Embedding function
    # -----------------------------------------------------------------------

    def get_embedding_func(self) -> Optional[Callable]:
        """Return an embedding callable for the configured model, or None.

        When ``embedding_model`` is set and the provider is Ollama,
        returns a ``(text: str) -> list[float]`` callable that calls
        the Ollama ``/api/embeddings`` endpoint.

        When ``embedding_model`` is empty, returns None (ChromaDB
        will use its built-in default model).
        """
        if not self.embedding_model:
            return None

        provider = self.llm_provider.lower()
        if provider == "ollama":
            return self._build_ollama_embedding_func()

        # Other providers could be added here (OpenAI embeddings, etc.)
        return None

    def _build_ollama_embedding_func(self) -> Callable:
        """Build an embedding function targeting Ollama /api/embeddings."""
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for Ollama embeddings. "
                "Install it with:  pip install httpx"
            )

        base_url = self.llm_base_url.rstrip("/")
        model = self.embedding_model
        client = httpx.Client(timeout=60.0)

        def embed(text: str) -> list:
            resp = client.post(
                f"{base_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("embedding", [])

        return embed

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (YAML/JSON-safe, no callables)."""
        return {
            "data_dir": str(self.data_dir),
            "core_person": self.core_person,
            "default_project": self.default_project,
            "signal_mode": self.signal_mode,
            "code_signal_mode": self.code_signal_mode,
            "extract_mode": self.extract_mode,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "llm_weight": self.llm_weight,
            "token_budget": self.token_budget,
            "decay_half_life_hours": self.decay_half_life_hours,
            "max_traces": self.max_traces,
            "reinforce_delta": self.reinforce_delta,
            "weaken_delta": self.weaken_delta,
            "identity_share": self.identity_share,
            "relationship_share": self.relationship_share,
            "grounding_share": self.grounding_share,
            "recent_conversation_share": self.recent_conversation_share,
            "episodic_share": self.episodic_share,
            "procedural_share": self.procedural_share,
            "reserve_share": self.reserve_share,
            "signal_health_threshold": self.signal_health_threshold,
            "reinforce_threshold": self.reinforce_threshold,
            "weaken_threshold": self.weaken_threshold,
            "compaction_keep_recent": self.compaction_keep_recent,
            "compaction_segment_size": self.compaction_segment_size,
            "compaction_min_messages": self.compaction_min_messages,
            "consolidation_min_episodes": self.consolidation_min_episodes,
            "consolidation_time_window_hours": self.consolidation_time_window_hours,
            "consolidation_min_threads": self.consolidation_min_threads,
            "consolidation_max_episodes_per_run": self.consolidation_max_episodes_per_run,
            # workspace
            "workspace_capacity": self.workspace_capacity,
            "workspace_decay_rate": self.workspace_decay_rate,
            "workspace_rehearsal_boost": self.workspace_rehearsal_boost,
            "workspace_expiry_threshold": self.workspace_expiry_threshold,
            # code-first boot
            "boot_n_sessions": self.boot_n_sessions,
            "boot_n_decisions": self.boot_n_decisions,
            # retrieval
            "embedding_model": self.embedding_model,
            "reranker_enabled": self.reranker_enabled,
            "reranker_model": self.reranker_model,
            "reranker_device": self.reranker_device,
        }
