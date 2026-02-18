"""
engram.search.reranker — Cross-encoder reranking for search results.

After hybrid search (FTS5 + semantic) produces candidate results via
Reciprocal Rank Fusion, a cross-encoder rescores each (query, document)
pair by jointly encoding them.  This is significantly more accurate than
bi-encoder similarity because the cross-encoder sees both texts together
and can model fine-grained relevance.

The default model (``cross-encoder/ms-marco-MiniLM-L-12-v2``) is:
  - ~33M parameters, runs on CPU in ~30ms for 30 documents
  - Trained on MS MARCO passage ranking (500k+ pairs)
  - NDCG@10 = 0.390 on MS MARCO dev

This is the single highest-impact RAG optimization per retrieval
research (Nogueira et al., 2020; Glass et al., 2022).

Dependencies:
  - ``sentence-transformers`` (optional — reranking is best-effort)
  - ``torch`` (pulled in by sentence-transformers)

If dependencies are missing, ``Reranker.rerank()`` returns results
unchanged (no-op fallback).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Default cross-encoder model.  Good balance of speed and accuracy.
# Alternatives:
#   - cross-encoder/ms-marco-TinyBERT-L-2-v2  (faster, less accurate)
#   - cross-encoder/ms-marco-MiniLM-L-6-v2    (middle ground)
#   - BAAI/bge-reranker-v2-m3                  (multilingual, larger)
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class Reranker:
    """Cross-encoder reranker with lazy model loading.

    The model is loaded on first call to ``rerank()``, not on
    construction.  This keeps import and init fast when reranking
    is not needed.

    Parameters
    ----------
    model_name:
        HuggingFace model ID for the cross-encoder.
    max_length:
        Maximum token length for input pairs (truncates longer docs).
    device:
        Torch device (``"cpu"``, ``"cuda"``, or ``None`` for auto).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 512,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._device = device
        self._model = None  # loaded lazily
        self._available: Optional[bool] = None  # None = not checked yet

    @property
    def available(self) -> bool:
        """Check if sentence-transformers is importable and functional."""
        if self._available is None:
            try:
                from sentence_transformers import CrossEncoder  # noqa: F401

                self._available = True
            except ImportError:
                self._available = False
                log.info(
                    "sentence-transformers not installed; "
                    "cross-encoder reranking disabled. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as exc:
                # Catch dependency conflicts (torch/torchvision version
                # mismatches, missing CUDA libs, etc.)
                self._available = False
                log.info(
                    "sentence-transformers import failed (%s); "
                    "cross-encoder reranking disabled.",
                    exc,
                )
        return self._available

    def _load_model(self) -> None:
        """Load the cross-encoder model (first-use only)."""
        if self._model is not None:
            return
        if not self.available:
            return

        from sentence_transformers import CrossEncoder

        log.info("Loading cross-encoder model: %s", self._model_name)
        self._model = CrossEncoder(
            self._model_name,
            max_length=self._max_length,
            device=self._device,
        )
        log.info("Cross-encoder loaded successfully")

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_n: Optional[int] = None,
        content_key: str = "content",
        score_key: str = "rerank_score",
    ) -> List[Dict]:
        """Rerank search results using the cross-encoder.

        Parameters
        ----------
        query:
            The original search query.
        results:
            List of search result dicts, each containing a text field.
        top_n:
            If set, only return the top N results after reranking.
            If None, returns all results re-sorted.
        content_key:
            Key in each result dict containing the text to score.
        score_key:
            Key to store the cross-encoder relevance score in each result.

        Returns
        -------
        list[dict]
            Results re-sorted by cross-encoder score (highest first).
            Each result has an additional ``score_key`` field with the
            cross-encoder relevance score.

        If the cross-encoder is unavailable (missing dependencies or
        failed to load), returns ``results`` unchanged.
        """
        if not results:
            return results

        if not self.available:
            return results

        # Lazy-load the model
        try:
            self._load_model()
        except Exception as exc:
            log.warning("Failed to load cross-encoder: %s", exc)
            return results

        if self._model is None:
            return results

        # Build (query, document) pairs for the cross-encoder
        pairs = []
        valid_indices = []
        for i, result in enumerate(results):
            text = result.get(content_key, "")
            if text:
                pairs.append((query, text))
                valid_indices.append(i)

        if not pairs:
            return results

        # Score all pairs in a single batch
        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            log.warning("Cross-encoder scoring failed: %s", exc)
            return results

        # Attach scores to results
        for idx, score in zip(valid_indices, scores):
            results[idx][score_key] = float(score)

        # Results without content keep their original position via
        # a very low rerank score
        for i, result in enumerate(results):
            if score_key not in result:
                result[score_key] = -999.0

        # Sort by cross-encoder score (highest = most relevant)
        results.sort(key=lambda r: r.get(score_key, -999.0), reverse=True)

        if top_n is not None:
            return results[:top_n]
        return results
