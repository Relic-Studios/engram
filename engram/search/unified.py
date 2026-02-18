"""
engram.search.unified — Unified search combining FTS5 + semantic.

Merges keyword (SQLite FTS5) and vector (ChromaDB) search results
using Reciprocal Rank Fusion (RRF), deduplicates, and returns a
combined ranking.

RRF replaces the previous 50/50 weighted-average approach.  It is
rank-based (not score-based) so it avoids the normalization asymmetry
between FTS5 local min-max and cosine global /2.0.  RRF is proven to
improve hybrid search precision by 15-25% and is used by Elasticsearch,
Weaviate, and Vespa in production.

Reference: Cormack, Clarke & Buettcher (2009) — "Reciprocal Rank Fusion
outperforms Condorcet and individual rank learning methods."
"""

from __future__ import annotations

from typing import Dict, List, Optional

from engram.search.indexed import IndexedSearch
from engram.search.reranker import Reranker
from engram.search.semantic import SemanticSearch


class UnifiedSearch:
    """Combined keyword + semantic search across all memory types.

    Pipeline: FTS5 + Semantic → RRF merge → Cross-encoder rerank

    If only ``indexed`` is provided, semantic search is skipped.
    This lets the system work without embeddings while still
    benefiting from them when available.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both
    sources, then optionally rescores the top candidates with a
    cross-encoder for higher-precision relevance ranking.
    """

    # Default cosine distance cutoff.  Cosine distance ranges from
    # 0 (identical) to 2 (opposite).  Results beyond this threshold
    # are too dissimilar to be useful and are discarded before merge.
    DEFAULT_SIMILARITY_THRESHOLD: float = 1.5

    # RRF smoothing constant.  Higher values reduce the advantage of
    # top-ranked results; 60 is the standard from the original paper.
    DEFAULT_RRF_K: int = 60

    def __init__(
        self,
        indexed: IndexedSearch,
        semantic: Optional[SemanticSearch] = None,
        similarity_threshold: Optional[float] = None,
        rrf_k: Optional[int] = None,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self.indexed = indexed
        self.semantic = semantic
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.DEFAULT_SIMILARITY_THRESHOLD
        )
        self.rrf_k = rrf_k if rrf_k is not None else self.DEFAULT_RRF_K
        self.reranker = reranker

    def search(
        self,
        query: str,
        person: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Run FTS + semantic search, merge, deduplicate, and rank.

        Parameters
        ----------
        query:
            Search query (natural language or keywords).
        person:
            Filter to this person (where applicable).
        memory_type:
            ``"messages"``, ``"traces"``, or ``None`` for both.
        limit:
            Maximum total results to return.

        Returns
        -------
        list[dict]
            Merged results, each with a ``"combined_score"`` key
            (higher = better match, via Reciprocal Rank Fusion).
        """
        if not query or not query.strip():
            return []

        # -- FTS5 search ---------------------------------------------------
        fts_results = self.indexed.search(
            query=query,
            memory_type=memory_type,
            person=person,
            limit=limit,
        )

        # -- Semantic search -----------------------------------------------
        sem_results: List[Dict] = []
        if self.semantic is not None:
            # Map memory_type to ChromaDB collections
            collections = self._map_collections(memory_type)
            where = None
            if person:
                where = {"person": person}

            try:
                sem_results = self.semantic.search(
                    query=query,
                    collections=collections,
                    n_results=limit,
                    where=where,
                )
                # FR-3: Filter out results above the similarity threshold.
                # Cosine distance [0, 2]: 0 = identical, 2 = opposite.
                sem_results = [
                    r
                    for r in sem_results
                    if r.get("distance", 2.0) <= self.similarity_threshold
                ]
            except Exception:
                # Semantic search is best-effort; don't fail the whole query
                sem_results = []

        # -- Merge via Reciprocal Rank Fusion ------------------------------
        merged = self._merge_rrf(fts_results, sem_results, k=self.rrf_k)

        # -- Sort by combined score (higher = better for RRF) --------------
        merged.sort(key=lambda r: r.get("combined_score", 0.0), reverse=True)

        # -- Cross-encoder reranking (optional) ----------------------------
        # Rerank the top candidates for higher-precision relevance.
        # The cross-encoder sees (query, document) jointly, which is
        # much more accurate than bi-encoder similarity or RRF alone.
        if self.reranker is not None:
            merged = self.reranker.rerank(
                query=query,
                results=merged,
                top_n=limit,
            )

        return merged[:limit]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _map_collections(memory_type: Optional[str]) -> Optional[List[str]]:
        """Map a ``memory_type`` filter to ChromaDB collection names.

        When ``memory_type`` is None, returns None which tells
        SemanticSearch to query all active collections (including
        the code collection if dual-embedding is enabled).
        """
        if memory_type == "messages":
            # Messages aren't in ChromaDB by default; search episodic
            # which contains traces from conversations
            return ["episodic"]
        if memory_type == "traces":
            # Search both NL and code embeddings for traces
            return ["episodic", "code"]
        # None → search all active collections
        return None

    @staticmethod
    def _merge_rrf(
        fts_results: List[Dict],
        sem_results: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        """Merge FTS and semantic results using Reciprocal Rank Fusion.

        RRF score for each document = sum over sources of:
            1 / (k + rank_in_source)

        where rank is 1-based (best result = rank 1).

        Documents appearing in both sources get contributions from
        both, naturally boosting items that both retrieval methods
        agree on.  Documents in only one source still participate
        with their single-source RRF score.

        Deduplication key priority:
        1. ``id`` field (messages/traces have a 12-char hex ID)
        2. ``doc_id`` field (semantic results)
        3. Content hash as fallback

        Parameters
        ----------
        k : int
            RRF smoothing constant (default 60, per Cormack et al.).
            Higher values reduce the advantage of top-ranked results.

        Returns
        -------
        list[dict]
            Merged results with ``combined_score`` (higher = better).
        """
        seen: Dict[str, Dict] = {}

        # -- Process FTS results (already sorted by rank, best first) ------
        for rank_0, result in enumerate(fts_results):
            rank = rank_0 + 1  # 1-based
            key = _dedup_key(result)
            rrf_score = 1.0 / (k + rank)

            if key in seen:
                seen[key]["combined_score"] += rrf_score
                seen[key]["fts_rank"] = rank
            else:
                entry = dict(result)
                entry["combined_score"] = rrf_score
                entry["fts_rank"] = rank
                entry["sem_rank"] = None
                seen[key] = entry

        # -- Process semantic results (already sorted by distance, best first)
        for rank_0, result in enumerate(sem_results):
            rank = rank_0 + 1  # 1-based
            key = _dedup_key(result)
            rrf_score = 1.0 / (k + rank)

            if key in seen:
                seen[key]["combined_score"] += rrf_score
                seen[key]["sem_rank"] = rank
                # Enrich with semantic metadata if missing
                if "collection" not in seen[key]:
                    seen[key]["collection"] = result.get("collection", "")
            else:
                entry = dict(result)
                entry["combined_score"] = rrf_score
                entry["sem_rank"] = rank
                entry["fts_rank"] = None
                # Normalise field names to match FTS format
                if "content" not in entry and "doc_id" in entry:
                    entry["content"] = result.get("content", "")
                seen[key] = entry

        return list(seen.values())


def _normalize_trace_id(raw_id: str) -> str:
    """Strip known prefixes to get the canonical trace ID.

    ChromaDB stores doc IDs as ``"trace_<id>"``, ``"skill_<hash>"``,
    ``"rel_<hash>"``, ``"soul_p<n>_<hash>"``, etc.  FTS results use
    the bare 12-char hex ID.  This normalises both to the same form
    so deduplication actually works (FR-4).
    """
    if raw_id.startswith("trace_"):
        return raw_id[6:]  # strip "trace_" prefix
    return raw_id


def _dedup_key(result: Dict) -> str:
    """Compute a deduplication key for a search result.

    Normalises ID formats so FTS results (``id="abc123def"``) and
    semantic results (``doc_id="trace_abc123def"``) match correctly.
    """
    # Extract the raw ID from whichever field is present
    raw_id = result.get("id") or result.get("trace_id") or result.get("doc_id") or ""

    if raw_id:
        return f"id:{_normalize_trace_id(raw_id)}"

    # Also check metadata — ChromaDB results carry trace_id in metadata
    meta = result.get("metadata", {})
    if isinstance(meta, dict) and meta.get("trace_id"):
        return f"id:{_normalize_trace_id(meta['trace_id'])}"

    # Fall back to content hash
    content = result.get("content", "")
    return f"hash:{hash(content)}"
