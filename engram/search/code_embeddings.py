"""
engram.search.code_embeddings — Dual-embedding engine for code-optimized retrieval.

Provides a code-specific embedding model (via sentence-transformers)
alongside the existing NL embedding model (via Ollama/ChromaDB default).

Code content is embedded with a model trained specifically on source code
(function signatures, imports, stack traces, etc.) while natural language
content continues using the general-purpose NL model.

Default code model: jinaai/jina-embeddings-v2-base-code
  - 768d embeddings, 8192 token context
  - Trained on GitHub + StackOverflow code-NL pairs
  - ~137M parameters, runs on CPU in ~50ms per document
  - Apache 2.0 license

If sentence-transformers or the model is unavailable, the system falls
back gracefully to NL-only embeddings (current behavior).

Architecture:
  - Code traces are dual-indexed: NL collection (episodic) + code collection
  - NL traces are indexed in existing collections only
  - At search time, all collections are queried, RRF merges results
  - The code collection uses code embeddings; others use NL embeddings
  - ChromaDB handles query embedding per-collection automatically
"""

from __future__ import annotations

import logging
import re
from typing import Callable, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace kind classification
# ---------------------------------------------------------------------------

#: Trace kinds that represent code content.  These traces get dual-indexed
#: in both the episodic (NL) and code collections.
CODE_TRACE_KINDS = frozenset(
    {
        "code_symbols",
        "code_pattern",
        "debug_session",
        "error_resolution",
        "test_strategy",
        "code_review",
        "wiring_map",
    }
)

# Patterns that strongly indicate code content (for heuristic classification
# when trace kind is not available or is generic like "episode").
_CODE_PATTERNS = re.compile(
    r"""
    (?:
        ^[ \t]*(?:def|class|import|from|async\s+def)\s   # Python
      | ^[ \t]*(?:function|const|let|var|export|import)\s  # JS/TS
      | ^[ \t]*(?:interface|type|enum)\s                   # TS
      | (?:=>)                                             # Arrow functions
      | (?:\w+\.\w+\.\w+)                                 # Dot chains (a.b.c)
      | (?:Traceback|Error:|Exception:|at\s+\w+\s*\()     # Stack traces
      | (?:```(?:python|javascript|typescript|js|ts|py))   # Fenced code blocks
    )
    """,
    re.MULTILINE | re.VERBOSE,
)


def is_code_content(
    trace_kind: Optional[str] = None,
    content: str = "",
) -> bool:
    """Classify whether content should be indexed in the code collection.

    Uses a two-level classifier:
    1. If ``trace_kind`` is in ``CODE_TRACE_KINDS``, returns True immediately.
    2. Falls back to regex pattern matching on content.

    Parameters
    ----------
    trace_kind:
        The trace kind (e.g., "code_symbols", "episode").
    content:
        The text content to classify.

    Returns
    -------
    bool
        True if this content should be indexed with code embeddings.
    """
    if trace_kind and trace_kind in CODE_TRACE_KINDS:
        return True
    if not content:
        return False
    return bool(_CODE_PATTERNS.search(content))


# ---------------------------------------------------------------------------
# Code embedding model
# ---------------------------------------------------------------------------

# Default model — best-in-class local code embedding model.
# 768d (matches nomic-embed-text), 8192 token context, code-specific.
DEFAULT_CODE_MODEL = "jinaai/jina-embeddings-v2-base-code"


class CodeEmbedder:
    """Lazy-loading code embedding model via sentence-transformers.

    The model is downloaded from HuggingFace on first use.  Subsequent
    calls reuse the loaded model.  If sentence-transformers or CUDA is
    unavailable, falls back to CPU or returns None.

    Parameters
    ----------
    model_name:
        HuggingFace model ID for the code embedding model.
    device:
        Torch device ("cpu", "cuda", or "" for auto-detect).
    max_length:
        Maximum token length for input (truncates longer texts).
        Default 8192 matches jina-embeddings-v2-base-code context.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CODE_MODEL,
        device: str = "",
        max_length: int = 8192,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._model = None
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        """Check if sentence-transformers is importable."""
        if self._available is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: F401

                self._available = True
            except ImportError:
                self._available = False
                log.info(
                    "sentence-transformers not installed; "
                    "code embeddings disabled. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as exc:
                self._available = False
                log.info(
                    "sentence-transformers import failed (%s); "
                    "code embeddings disabled.",
                    exc,
                )
        return self._available

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> Optional[int]:
        """Embedding dimension, or None if model not loaded."""
        if self._model is None:
            return None
        try:
            return self._model.get_sentence_embedding_dimension()
        except Exception:
            return None

    def _load_model(self) -> None:
        """Load the embedding model (first-use only)."""
        if self._model is not None:
            return
        if not self.available:
            return

        from sentence_transformers import SentenceTransformer

        # Auto-detect device
        device = self._device
        if not device:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        log.info(
            "Loading code embedding model: %s (device=%s)",
            self._model_name,
            device,
        )
        try:
            self._model = SentenceTransformer(
                self._model_name,
                device=device,
                trust_remote_code=True,
            )
            log.info(
                "Code embedding model loaded: %dd embeddings",
                self._model.get_sentence_embedding_dimension(),
            )
        except Exception as exc:
            log.warning("Failed to load code embedding model: %s", exc)
            self._available = False

    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> Optional[List[List[float]]]:
        """Encode a batch of texts into embeddings.

        Parameters
        ----------
        texts:
            List of texts to embed.
        normalize:
            Whether to L2-normalize embeddings (recommended for cosine
            similarity).

        Returns
        -------
        list[list[float]] or None
            Embedding vectors, or None if model is unavailable.
        """
        if not texts:
            return []

        try:
            self._load_model()
        except Exception as exc:
            log.warning("Code embedding encode failed: %s", exc)
            return None

        if self._model is None:
            return None

        try:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            # Convert numpy array to list of lists
            return [emb.tolist() for emb in embeddings]
        except Exception as exc:
            log.warning("Code embedding encode failed: %s", exc)
            return None

    def encode_single(self, text: str) -> Optional[List[float]]:
        """Encode a single text.  Convenience wrapper around ``encode()``."""
        result = self.encode([text])
        if result is None or len(result) == 0:
            return None
        return result[0]


# ---------------------------------------------------------------------------
# Factory function for Config integration
# ---------------------------------------------------------------------------


def build_code_embedding_func(
    model_name: str = DEFAULT_CODE_MODEL,
    device: str = "",
) -> Optional[Callable]:
    """Build a ``(text: str) -> list[float]`` callable for code embeddings.

    Returns None if sentence-transformers is not available.

    The returned callable loads the model lazily on first invocation.
    This keeps startup fast — the ~500MB model is only downloaded when
    code embedding is first needed.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.
    device:
        Torch device ("cpu", "cuda", or "" for auto-detect).

    Returns
    -------
    callable or None
        Embedding function, or None if deps are missing.
    """
    embedder = CodeEmbedder(model_name=model_name, device=device)
    if not embedder.available:
        return None

    def embed(text: str) -> list:
        result = embedder.encode_single(text)
        if result is None:
            raise RuntimeError(
                f"Code embedding model {model_name!r} failed to encode. "
                "Check logs for details."
            )
        return result

    return embed
