"""
engram.consolidation.consolidator — Hierarchical memory consolidation.

Groups individual episodic traces into **threads** (topical clusters)
and threads into **arcs** (long-term narrative themes). Inspired by:

  - Neuroscience: hippocampal replay consolidates episodes into schemas
  - NotebookLM: auto-generated "Source Guides" synthesise facts
  - MemGPT: agent can trigger its own reflection/consolidation

How it works:
  1. **Thread creation**: Fetch recent episode/summary traces, group by
     person-tag co-occurrence + temporal proximity, then summarise each
     cluster into a ``kind="thread"`` trace.
  2. **Arc creation**: Fetch existing threads, group by shared people or
     tag overlap, then synthesise each group into a ``kind="arc"`` trace.

Original traces are NEVER deleted — threads and arcs are strictly
additive, higher-abstraction summaries that link back to their children
via ``metadata.child_ids``.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import hdbscan as _hdbscan

    _HAS_HDBSCAN = True
except ImportError:
    _HAS_HDBSCAN = False

if TYPE_CHECKING:
    from engram.core.types import LLMFunc
    from engram.episodic.store import EpisodicStore
    from engram.search.semantic import SemanticSearch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_THREAD_SYSTEM = (
    "You are a memory-consolidation system. Synthesise a cluster of "
    "episodic traces into a single cohesive **thread** summary written "
    "in first person. Preserve: key facts, emotional tone, decisions, "
    "and anything that changed the relationship. Be concise — under 150 "
    "words. Start with the main theme of this thread."
)

_ARC_SYSTEM = (
    "You are a memory-consolidation system. Synthesise several related "
    "threads into a long-term **arc** — a narrative summary of how a "
    "relationship or theme has evolved over time. Write in first person. "
    "Focus on growth, turning points, and the current state. Under 200 "
    "words."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class MemoryConsolidator:
    """Hierarchical consolidation: episodes → threads → arcs.

    Parameters
    ----------
    min_episodes_per_thread : int
        Minimum episode traces in a cluster before creating a thread.
    thread_time_window_hours : float
        Episodes within this time window (and sharing a person tag)
        are candidates for the same thread.
    min_threads_per_arc : int
        Minimum threads before consolidating into an arc.
    max_episodes_per_run : int
        Limit on how many episode traces to consider per consolidation
        run (prevents runaway on huge stores).
    semantic_search : SemanticSearch | None
        Optional ChromaDB semantic search instance.  When provided
        (and ``hdbscan`` is installed), topic-coherent clustering
        replaces pure temporal clustering: HDBSCAN groups traces by
        embedding similarity, then temporal sub-splitting is applied
        within each topic cluster.  Falls back to temporal-only
        clustering when unavailable.
    """

    def __init__(
        self,
        min_episodes_per_thread: int = 5,
        thread_time_window_hours: float = 72.0,
        min_threads_per_arc: int = 3,
        max_episodes_per_run: int = 200,
        semantic_search: Optional["SemanticSearch"] = None,
    ) -> None:
        self.min_episodes_per_thread = min_episodes_per_thread
        self.thread_time_window_hours = thread_time_window_hours
        self.min_threads_per_arc = min_threads_per_arc
        self.max_episodes_per_run = max_episodes_per_run
        self._semantic_search = semantic_search

    # ------------------------------------------------------------------
    # Thread consolidation
    # ------------------------------------------------------------------

    def consolidate_threads(
        self,
        episodic: EpisodicStore,
        llm_func: Optional[LLMFunc] = None,
    ) -> List[str]:
        """Group un-threaded episodes into threads.

        Returns a list of newly created thread trace IDs.
        """
        # Fetch episode/summary traces that haven't been consolidated
        episodes = _get_unconsolidated_episodes(
            episodic, limit=self.max_episodes_per_run
        )
        if len(episodes) < self.min_episodes_per_thread:
            log.debug(
                "Only %d unconsolidated episodes — below threshold %d",
                len(episodes),
                self.min_episodes_per_thread,
            )
            return []

        # Group by person tag
        person_groups = _group_by_person(episodes)

        created_thread_ids: List[str] = []

        for person, person_episodes in person_groups.items():
            # Topic-coherent clustering (HDBSCAN on embeddings)
            # with temporal sub-splitting, falling back to
            # temporal-only when embeddings aren't available.
            clusters = _cluster_by_topic(
                person_episodes,
                self.thread_time_window_hours,
                self.min_episodes_per_thread,
                self._semantic_search,
            )

            for cluster in clusters:
                if len(cluster) < self.min_episodes_per_thread:
                    continue

                thread_id = self._create_thread(
                    person=person,
                    episodes=cluster,
                    episodic=episodic,
                    llm_func=llm_func,
                )
                if thread_id:
                    created_thread_ids.append(thread_id)

        if created_thread_ids:
            log.info(
                "Thread consolidation: created %d threads", len(created_thread_ids)
            )
        return created_thread_ids

    # ------------------------------------------------------------------
    # Arc consolidation
    # ------------------------------------------------------------------

    def consolidate_arcs(
        self,
        episodic: EpisodicStore,
        llm_func: Optional[LLMFunc] = None,
    ) -> List[str]:
        """Group un-arced threads into arcs.

        Returns a list of newly created arc trace IDs.
        """
        threads = _get_unconsolidated_threads(episodic)
        if len(threads) < self.min_threads_per_arc:
            log.debug(
                "Only %d unconsolidated threads — below threshold %d",
                len(threads),
                self.min_threads_per_arc,
            )
            return []

        # Group threads by person
        person_groups = _group_by_person(threads)

        created_arc_ids: List[str] = []

        for person, person_threads in person_groups.items():
            if len(person_threads) < self.min_threads_per_arc:
                continue

            arc_id = self._create_arc(
                person=person,
                threads=person_threads,
                episodic=episodic,
                llm_func=llm_func,
            )
            if arc_id:
                created_arc_ids.append(arc_id)

        if created_arc_ids:
            log.info("Arc consolidation: created %d arcs", len(created_arc_ids))
        return created_arc_ids

    # ------------------------------------------------------------------
    # Full consolidation pass
    # ------------------------------------------------------------------

    def consolidate(
        self,
        episodic: EpisodicStore,
        llm_func: Optional[LLMFunc] = None,
    ) -> Dict[str, List[str]]:
        """Run both thread and arc consolidation.

        Returns ``{"threads": [...ids], "arcs": [...ids]}``.
        """
        thread_ids = self.consolidate_threads(episodic, llm_func)
        arc_ids = self.consolidate_arcs(episodic, llm_func)
        return {"threads": thread_ids, "arcs": arc_ids}

    # ------------------------------------------------------------------
    # Internal: create individual thread/arc
    # ------------------------------------------------------------------

    def _create_thread(
        self,
        person: str,
        episodes: List[Dict],
        episodic: EpisodicStore,
        llm_func: Optional[LLMFunc],
    ) -> Optional[str]:
        """Summarise a cluster of episodes into a thread trace."""
        child_ids = [e["id"] for e in episodes if e.get("id")]
        episode_text = _format_traces_for_llm(episodes)

        # Summarise
        if llm_func is not None:
            summary = _llm_summarise(episode_text, _THREAD_SYSTEM, llm_func)
        else:
            summary = _extractive_thread_summary(episodes, person)

        if not summary:
            log.warning(
                "Empty thread summary for %s (%d episodes), skipping",
                person,
                len(episodes),
            )
            return None

        # Derive time range
        time_range = _time_range(episodes)

        try:
            thread_id = episodic.log_trace(
                content=summary,
                kind="thread",
                tags=[person, "consolidation"],
                salience=0.75,  # threads are more salient than individual episodes
                child_ids=child_ids,
                time_range=time_range,
                episode_count=len(episodes),
            )

            # Mark child episodes as consolidated
            _mark_consolidated(episodic, child_ids, thread_id)

            log.debug(
                "Created thread %s for %s (%d episodes, %s)",
                thread_id,
                person,
                len(episodes),
                time_range,
            )
            return thread_id

        except Exception as exc:
            log.warning("Failed to create thread trace: %s", exc)
            return None

    def _create_arc(
        self,
        person: str,
        threads: List[Dict],
        episodic: EpisodicStore,
        llm_func: Optional[LLMFunc],
    ) -> Optional[str]:
        """Synthesise multiple threads into an arc trace."""
        child_ids = [t["id"] for t in threads if t.get("id")]
        thread_text = _format_traces_for_llm(threads)

        if llm_func is not None:
            summary = _llm_summarise(thread_text, _ARC_SYSTEM, llm_func)
        else:
            summary = _extractive_arc_summary(threads, person)

        if not summary:
            log.warning(
                "Empty arc summary for %s (%d threads), skipping",
                person,
                len(threads),
            )
            return None

        time_range = _time_range(threads)

        try:
            arc_id = episodic.log_trace(
                content=summary,
                kind="arc",
                tags=[person, "consolidation"],
                salience=0.85,  # arcs are the highest-salience summaries
                child_thread_ids=child_ids,
                time_range=time_range,
                thread_count=len(threads),
            )

            # Mark child threads as consolidated
            _mark_consolidated(episodic, child_ids, arc_id)

            log.debug(
                "Created arc %s for %s (%d threads, %s)",
                arc_id,
                person,
                len(threads),
                time_range,
            )
            return arc_id

        except Exception as exc:
            log.warning("Failed to create arc trace: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Private helpers — querying
# ---------------------------------------------------------------------------


def _get_unconsolidated_episodes(
    episodic: EpisodicStore, limit: int = 200
) -> List[Dict]:
    """Get episode/summary traces that haven't been rolled into a thread yet.

    A trace is "unconsolidated" if its metadata doesn't contain a
    ``consolidated_into`` key.
    """
    rows = episodic.conn.execute(
        """
        SELECT * FROM traces
        WHERE kind IN ('episode', 'summary')
          AND COALESCE(json_extract(metadata, '$.consolidated_into'), '') = ''
        ORDER BY created ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [episodic._row_to_dict(r) for r in rows]


def _get_unconsolidated_threads(episodic: EpisodicStore) -> List[Dict]:
    """Get thread traces not yet rolled into an arc."""
    rows = episodic.conn.execute(
        """
        SELECT * FROM traces
        WHERE kind = 'thread'
          AND COALESCE(json_extract(metadata, '$.consolidated_into'), '') = ''
        ORDER BY created ASC
        """,
    ).fetchall()
    return [episodic._row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Private helpers — grouping
# ---------------------------------------------------------------------------


def _group_by_person(traces: List[Dict]) -> Dict[str, List[Dict]]:
    """Group traces by the first person-like tag (not 'consolidation', etc.)."""
    skip_tags = {"consolidation", "compaction", "test"}
    groups: Dict[str, List[Dict]] = defaultdict(list)

    for trace in traces:
        tags = trace.get("tags") or []
        person = None
        for tag in tags:
            if tag.lower() not in skip_tags:
                person = tag
                break
        if person is None:
            person = "_unknown"
        groups[person].append(trace)

    return dict(groups)


def _cluster_by_topic(
    traces: List[Dict],
    window_hours: float,
    min_cluster_size: int,
    semantic_search: Optional["SemanticSearch"],
) -> List[List[Dict]]:
    """Topic-coherent clustering with temporal sub-splitting.

    Algorithm (when HDBSCAN + embeddings are available):
      1. Fetch pre-computed embeddings from ChromaDB for each trace.
      2. Run HDBSCAN (density-based, no k required) on the embedding
         matrix.  ``min_cluster_size`` maps to the consolidation
         threshold.
      3. Within each semantic cluster, apply temporal proximity
         splitting (the existing 72h-window gap-break).
      4. Noise points (HDBSCAN label -1) are clustered temporally
         as a fallback group.

    Falls back to ``_cluster_by_time()`` when:
      - ``hdbscan`` package is not installed
      - ``semantic_search`` is ``None``
      - No embeddings could be retrieved
      - Fewer than ``min_cluster_size`` traces have embeddings
    """
    if not traces:
        return []

    # Fast path: fall back to temporal-only when HDBSCAN isn't available
    if not _HAS_HDBSCAN or semantic_search is None:
        return _cluster_by_time(traces, window_hours)

    # Fetch embeddings from ChromaDB (traces are stored with "trace_" prefix)
    trace_ids = [t.get("id", "") for t in traces if t.get("id")]
    chroma_ids = [f"trace_{tid}" for tid in trace_ids]

    try:
        raw_embs = semantic_search.get_embeddings(chroma_ids, collection="episodic")
    except Exception as exc:
        log.debug("Failed to fetch embeddings for topic clustering: %s", exc)
        return _cluster_by_time(traces, window_hours)

    if not raw_embs or len(raw_embs) < min_cluster_size:
        log.debug(
            "Only %d embeddings available (need %d), falling back to temporal",
            len(raw_embs),
            min_cluster_size,
        )
        return _cluster_by_time(traces, window_hours)

    # Map back to bare trace IDs → embedding vectors
    emb_map = {k.removeprefix("trace_"): v for k, v in raw_embs.items()}

    # Split traces into those with and without embeddings
    traces_with_emb = [t for t in traces if t.get("id") in emb_map]
    traces_without_emb = [t for t in traces if t.get("id") not in emb_map]

    if len(traces_with_emb) < min_cluster_size:
        return _cluster_by_time(traces, window_hours)

    # Build embedding matrix (preserving order)
    embedding_matrix = np.array(
        [emb_map[t["id"]] for t in traces_with_emb], dtype=np.float32
    )

    # Run HDBSCAN — density-based clustering, handles noise
    try:
        clusterer = _hdbscan.HDBSCAN(
            min_cluster_size=max(2, min_cluster_size),
            min_samples=2,
            metric="euclidean",
            cluster_selection_method="eom",  # excess of mass (default)
        )
        labels = clusterer.fit_predict(embedding_matrix)
    except Exception as exc:
        log.warning("HDBSCAN clustering failed: %s", exc)
        return _cluster_by_time(traces, window_hours)

    # Group traces by HDBSCAN cluster label
    topic_groups: Dict[int, List[Dict]] = defaultdict(list)
    for trace, label in zip(traces_with_emb, labels):
        topic_groups[int(label)].append(trace)

    # Apply temporal sub-splitting within each topic cluster
    all_clusters: List[List[Dict]] = []

    for label, group in sorted(topic_groups.items()):
        if label == -1:
            # Noise points — cluster temporally as fallback
            temporal_clusters = _cluster_by_time(group, window_hours)
            all_clusters.extend(temporal_clusters)
        else:
            # Semantic cluster — apply temporal sub-splitting to catch
            # cases where the same topic spans disconnected time periods
            temporal_clusters = _cluster_by_time(group, window_hours)
            all_clusters.extend(temporal_clusters)

    # Also cluster traces without embeddings temporally
    if traces_without_emb:
        temporal_clusters = _cluster_by_time(traces_without_emb, window_hours)
        all_clusters.extend(temporal_clusters)

    log.info(
        "Topic-coherent clustering: %d traces → %d HDBSCAN clusters "
        "(%d noise) → %d final clusters after temporal split",
        len(traces_with_emb),
        len([l for l in set(labels) if l >= 0]),
        sum(1 for l in labels if l == -1),
        len(all_clusters),
    )

    return all_clusters


def _cluster_by_time(traces: List[Dict], window_hours: float) -> List[List[Dict]]:
    """Split traces into clusters where consecutive traces are within
    ``window_hours`` of each other.

    Input traces should be sorted by creation time (ASC).
    """
    if not traces:
        return []

    # Sort by created timestamp (should already be sorted, but be safe)
    sorted_traces = sorted(traces, key=lambda t: t.get("created", ""))

    clusters: List[List[Dict]] = []
    current: List[Dict] = [sorted_traces[0]]

    for i in range(1, len(sorted_traces)):
        gap = _time_gap_hours_trace(current[-1], sorted_traces[i])
        if gap > window_hours:
            clusters.append(current)
            current = [sorted_traces[i]]
        else:
            current.append(sorted_traces[i])

    if current:
        clusters.append(current)

    return clusters


def _time_gap_hours_trace(trace_a: Dict, trace_b: Dict) -> float:
    """Compute the time gap in hours between two traces using ``created``."""
    try:
        ts_a = trace_a.get("created", "")
        ts_b = trace_b.get("created", "")
        dt_a = datetime.fromisoformat(ts_a.replace("Z", "+00:00"))
        dt_b = datetime.fromisoformat(ts_b.replace("Z", "+00:00"))
        return abs((dt_b - dt_a).total_seconds()) / 3600.0
    except (ValueError, TypeError, AttributeError):
        return 0.0


# ---------------------------------------------------------------------------
# Private helpers — formatting for LLM
# ---------------------------------------------------------------------------


def _format_traces_for_llm(traces: List[Dict]) -> str:
    """Format a list of traces into readable text for the LLM summariser."""
    lines = []
    for i, trace in enumerate(traces, 1):
        kind = trace.get("kind", "episode")
        content = trace.get("content", "")
        created = trace.get("created", "")
        sal = trace.get("salience", 0.0)
        lines.append(f"[{i}] ({kind}, sal:{sal:.2f}, {created})\n  {content}")
    text = "\n\n".join(lines)
    # Protect against absurdly long input
    if len(text) > 15_000:
        text = text[:15_000] + "\n\n[...truncated]"
    return text


# ---------------------------------------------------------------------------
# Private helpers — summarisation
# ---------------------------------------------------------------------------


def _llm_summarise(text: str, system: str, llm_func: "LLMFunc") -> str:
    """Summarise text using the LLM with a given system prompt."""
    prompt = f"Consolidate these memory traces:\n\n{text}"
    try:
        return llm_func(prompt, system).strip()
    except Exception as exc:
        log.warning("LLM consolidation failed: %s", exc)
        return ""


def _extractive_thread_summary(episodes: List[Dict], person: str) -> str:
    """Fallback: pick highest-salience episodes and concatenate."""
    ranked = sorted(episodes, key=lambda e: e.get("salience", 0.0), reverse=True)
    top = ranked[: min(5, len(ranked))]
    top.sort(key=lambda e: e.get("created", ""))

    lines = []
    for ep in top:
        content = ep.get("content", "")
        if len(content) > 250:
            content = content[:250] + "..."
        lines.append(f"- {content}")

    time_range = _time_range(episodes)
    header = f"Thread with {person} ({time_range}):"
    return header + "\n" + "\n".join(lines)


def _extractive_arc_summary(threads: List[Dict], person: str) -> str:
    """Fallback: concatenate thread summaries chronologically."""
    sorted_threads = sorted(threads, key=lambda t: t.get("created", ""))
    lines = []
    for t in sorted_threads:
        content = t.get("content", "")
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"- {content}")

    time_range = _time_range(threads)
    header = f"Arc with {person} ({time_range}):"
    return header + "\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers — metadata management
# ---------------------------------------------------------------------------


def _time_range(traces: List[Dict]) -> str:
    """Return 'first_ts to last_ts' from a list of traces."""
    if not traces:
        return ""
    times = [t.get("created", "") for t in traces if t.get("created")]
    if not times:
        return ""
    times.sort()
    return f"{times[0]} to {times[-1]}"


def _mark_consolidated(
    episodic: EpisodicStore, trace_ids: List[str], parent_id: str
) -> None:
    """Set ``metadata.consolidated_into`` on each child trace.

    This prevents them from being re-consolidated in future runs.
    """
    for tid in trace_ids:
        try:
            row = episodic.conn.execute(
                "SELECT metadata FROM traces WHERE id = ?", (tid,)
            ).fetchone()
            if row is None:
                continue

            metadata: Dict = {}
            if row[0]:
                try:
                    metadata = json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

            metadata["consolidated_into"] = parent_id

            episodic.conn.execute(
                "UPDATE traces SET metadata = ? WHERE id = ?",
                (json.dumps(metadata), tid),
            )
        except Exception as exc:
            log.debug("Failed to mark trace %s as consolidated: %s", tid, exc)

    try:
        episodic.conn.commit()
    except Exception as exc:
        log.debug("Commit failed after marking consolidation: %s", exc)
