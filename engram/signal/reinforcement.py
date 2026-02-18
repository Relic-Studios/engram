"""
engram.signal.reinforcement — Citation-primary Hebbian reinforcement.

Strengthens or weakens memory traces based on:
  1. **Citation** (primary signal): Traces explicitly cited in the
     response (``[N]`` references) are proven useful and get full
     reinforcement.  Uncited traces get minimal reinforcement.
  2. **Signal health** (secondary signal): Controls whether
     reinforcement or weakening occurs.

This citation-primary approach replaces the original blanket
reinforcement where ALL context traces received equal deltas.
Research shows that citation-based feedback produces more
accurate salience tracking over time.

Reference: NotebookLM citation patterns; Asai et al. (2023)
"Self-RAG: Learning to Retrieve, Generate, and Critique."
"""

from __future__ import annotations

import logging
from typing import List, Set, Any

log = logging.getLogger(__name__)


class ReinforcementEngine:
    """
    Citation-primary Hebbian reinforcement for episodic memory.

    Reinforcement is proportional to evidence of usefulness:

    +-----------+-----------------+--------------------+
    | Health    | Cited traces    | Uncited traces     |
    +===========+=================+====================+
    | High      | +reinforce_delta| +context_delta     |
    | Dead band | (no change)     | (no change)        |
    | Low       | (protected)     | -weaken_delta      |
    +-----------+-----------------+--------------------+

    Key changes from blanket reinforcement:
    - Cited traces get full delta; uncited get only ``context_delta``
      (default 20% of ``reinforce_delta``).
    - Cited traces are NEVER weakened — citation proves value regardless
      of overall signal health.
    - Uncited traces are weakened on low signal, as they may have been
      retrieved but weren't useful.

    Parameters
    ----------
    reinforce_delta : float
        Amount added for *cited* traces on high signal (default 0.05).
    weaken_delta : float
        Amount subtracted for *uncited* traces on low signal (default 0.03).
    context_delta_ratio : float
        Fraction of ``reinforce_delta`` given to uncited-but-in-context
        traces on high signal (default 0.2 → +0.01).
    reinforce_threshold : float
        Signal health above which reinforcement occurs (default 0.7).
    weaken_threshold : float
        Signal health below which weakening occurs (default 0.4).
    """

    def __init__(
        self,
        reinforce_delta: float = 0.05,
        weaken_delta: float = 0.03,
        context_delta_ratio: float = 0.2,
        reinforce_threshold: float = 0.7,
        weaken_threshold: float = 0.4,
    ) -> None:
        self.reinforce_delta = reinforce_delta
        self.weaken_delta = weaken_delta
        self.context_delta_ratio = context_delta_ratio
        self.reinforce_threshold = reinforce_threshold
        self.weaken_threshold = weaken_threshold

    @property
    def context_delta(self) -> float:
        """Small reinforcement for uncited-but-in-context traces."""
        return self.reinforce_delta * self.context_delta_ratio

    def process(
        self,
        trace_ids: List[str],
        signal_health: float,
        episodic_store: Any,
        cited_ids: Set[str] | None = None,
    ) -> int:
        """
        Adjust salience of traces using citation-primary reinforcement.

        Parameters
        ----------
        trace_ids : list[str]
            IDs of all traces loaded into context for this exchange.
        signal_health : float
            Overall health score (0-1) from signal measurement.
        episodic_store
            Object with ``reinforce(table, id, delta)`` and
            ``weaken(table, id, delta)`` methods.
        cited_ids : set[str] | None
            IDs of traces that were cited in the response.
            If None, falls back to blanket reinforcement (backward compat).

        Returns
        -------
        int
            Number of traces adjusted.
        """
        if not trace_ids:
            return 0

        cited = cited_ids or set()
        adjusted = 0

        if signal_health > self.reinforce_threshold:
            for tid in trace_ids:
                try:
                    if tid in cited:
                        # Cited: full reinforcement — proven useful
                        episodic_store.reinforce("traces", tid, self.reinforce_delta)
                    else:
                        # Uncited but in context: minimal reinforcement
                        episodic_store.reinforce("traces", tid, self.context_delta)
                    adjusted += 1
                except Exception as exc:
                    log.debug("Failed to reinforce trace %s: %s", tid, exc)

        elif signal_health < self.weaken_threshold:
            for tid in trace_ids:
                # Cited traces are NEVER weakened — citation proves value
                if tid in cited:
                    continue
                try:
                    episodic_store.weaken("traces", tid, self.weaken_delta)
                    adjusted += 1
                except Exception as exc:
                    log.debug("Failed to weaken trace %s: %s", tid, exc)

        # else: dead band — no adjustment

        if cited:
            log.debug(
                "Citation-primary reinforcement: %d cited, %d uncited, "
                "health=%.2f, adjusted=%d",
                len(cited),
                len(trace_ids) - len(cited),
                signal_health,
                adjusted,
            )

        return adjusted
