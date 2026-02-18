"""
engram.working.allocator — Token budget allocator.

Greedy knapsack for fitting memories into a token budget,
text compression, message fitting, and lost-in-the-middle
reordering utilities.

The "Lost in the Middle" problem (Liu et al., 2023, Stanford/UCB):
LLMs attend more to information at the beginning and end of their
context window, with a U-shaped attention curve.  The ``reorder_u``
function places the most important items at the start and end.

**MMR diversity** (Carbonell & Goldstein, 1998):
Maximal Marginal Relevance penalises candidates that are too similar
to already-selected items.  This prevents redundant near-duplicate
traces from consuming the token budget.

    mmr_score(d) = lambda * relevance(d) - (1-lambda) * max_sim(d, selected)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from engram.core.tokens import estimate_tokens


# ── Cosine similarity helpers ────────────────────────────────


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors.  Returns 0.0 on degenerate input."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _max_sim_to_selected(
    candidate_emb: List[float],
    selected_embs: List[List[float]],
) -> float:
    """Maximum cosine similarity between *candidate_emb* and any selected embedding."""
    if not selected_embs:
        return 0.0
    return max(_cosine_similarity(candidate_emb, s) for s in selected_embs)


# ── Knapsack allocator ───────────────────────────────────────


def knapsack_allocate(
    items: List[Dict],
    budget: int,
    key_field: str = "salience",
    text_field: str = "content",
    embeddings: Optional[Dict[str, List[float]]] = None,
    id_field: str = "id",
    mmr_lambda: float = 0.7,
) -> Tuple[List[Dict], int]:
    """Greedy knapsack allocation with optional MMR diversity.

    Returns ``(selected_items, tokens_used)``.

    When *embeddings* is ``None`` (default), items are selected
    highest-density-first until the budget is exhausted — the
    original greedy algorithm.

    When *embeddings* is provided (a mapping ``{item_id: vector}``),
    the allocator applies **Maximal Marginal Relevance** (MMR) at
    each selection step:

        mmr_score = lambda * density - (1-lambda) * max_sim(d, selected)

    This penalises near-duplicate traces, promoting diversity in the
    selected set.  *mmr_lambda* controls the trade-off (0.7 = 70%
    relevance, 30% diversity is the recommended default).

    Parameters
    ----------
    items : list[dict]
        Candidate items for selection.
    budget : int
        Maximum token budget.
    key_field : str
        Dict key holding the relevance score (default ``"salience"``).
    text_field : str
        Dict key holding the text content (default ``"content"``).
    embeddings : dict[str, list[float]] | None
        Optional mapping of item IDs to embedding vectors.
        When provided, MMR diversity is applied.
    id_field : str
        Dict key holding the item's unique identifier (default ``"id"``).
    mmr_lambda : float
        Trade-off between relevance and diversity (default 0.7).
        1.0 = pure relevance (no diversity), 0.0 = pure diversity.
    """
    if not items or budget <= 0:
        return [], 0

    # Score each item by density = salience / tokens
    scored: List[Tuple[float, int, Dict]] = []
    for item in items:
        salience = item.get(key_field, 0.0)
        text = item.get(text_field, "")
        tokens = estimate_tokens(text)
        if tokens <= 0:
            continue
        density = salience / tokens
        scored.append((density, tokens, item))

    # Sort descending by density
    scored.sort(key=lambda x: x[0], reverse=True)

    # ── Fast path: no embeddings → original greedy ────────────
    use_mmr = embeddings is not None and len(embeddings) > 0 and mmr_lambda < 1.0

    if not use_mmr:
        selected: List[Dict] = []
        tokens_used = 0
        for _density, tokens, item in scored:
            if tokens_used + tokens > budget:
                continue
            selected.append(item)
            tokens_used += tokens
        return selected, tokens_used

    # ── MMR path ──────────────────────────────────────────────
    # Normalise densities to [0, 1] for comparable MMR scores.
    max_density = scored[0][0] if scored else 1.0
    if max_density < 1e-12:
        max_density = 1.0

    # Build candidate list with normalised density
    candidates = [
        (density / max_density, tokens, item) for density, tokens, item in scored
    ]

    selected: List[Dict] = []
    selected_embs: List[List[float]] = []
    tokens_used = 0

    while candidates:
        best_idx = -1
        best_score = -float("inf")

        for i, (norm_density, tokens, item) in enumerate(candidates):
            if tokens_used + tokens > budget:
                continue

            item_id = item.get(id_field, "")
            emb = embeddings.get(item_id) if item_id else None

            if emb is not None and selected_embs:
                sim = _max_sim_to_selected(emb, selected_embs)
            else:
                sim = 0.0

            mmr_score = mmr_lambda * norm_density - (1.0 - mmr_lambda) * sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx < 0:
            break  # nothing fits

        _, tokens, item = candidates.pop(best_idx)
        selected.append(item)
        tokens_used += tokens

        # Track the selected item's embedding for future similarity
        item_id = item.get(id_field, "")
        emb = embeddings.get(item_id) if item_id else None
        if emb is not None:
            selected_embs.append(emb)

    return selected, tokens_used


def compress_text(text: str, max_tokens: int) -> str:
    """Truncate *text* to fit within *max_tokens*.

    Attempts to cut at a sentence boundary (period, exclamation, or
    question mark followed by whitespace).  Appends ``[...truncated]``
    when text is actually cut.
    """
    if not text:
        return text

    current_tokens = estimate_tokens(text)
    if current_tokens <= max_tokens:
        return text

    suffix = " [...truncated]"
    # Target char count minus room for the suffix
    target_chars = max(0, max_tokens * 4 - len(suffix))
    truncated = text[:target_chars]

    # Try to cut at last sentence boundary
    for sep in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        last_idx = truncated.rfind(sep)
        if last_idx > len(truncated) // 2:  # don't cut too aggressively
            truncated = truncated[: last_idx + 1]
            break

    return truncated + suffix


def fit_messages(messages: List[Dict], budget: int) -> List[Dict]:
    """Fit the most recent messages within a token budget.

    Works backward from the most recent message, accumulating
    until the budget is exhausted.  Returns messages in
    chronological order.
    """
    if not messages or budget <= 0:
        return []

    selected: List[Dict] = []
    tokens_used = 0

    # Walk backwards (most recent first)
    for msg in reversed(messages):
        content = msg.get("content", "")
        tokens = estimate_tokens(content) + 4  # per-message overhead
        if tokens_used + tokens > budget:
            break
        selected.append(msg)
        tokens_used += tokens

    # Restore chronological order
    selected.reverse()
    return selected


def reorder_u(
    items: List[Dict],
    key_field: str = "salience",
) -> List[Dict]:
    """Reorder items in a U-shaped pattern for LLM attention.

    Given items sorted by relevance (best first), redistributes them
    so the most important items sit at the beginning and end of the
    list, with least important items in the middle.

    This counteracts the "Lost in the Middle" problem where LLMs
    attend poorly to middle-context items (Liu et al., 2023).

    Algorithm:
        1. Sort by relevance (if not already sorted)
        2. Alternate: assign item to front, then back, then front...
        3. Most relevant → position 0, 2nd → last, 3rd → position 1...

    Parameters
    ----------
    items:
        List of dicts, each having a ``key_field`` for relevance.
    key_field:
        Field to sort by (higher = more important).

    Returns
    -------
    list[dict]
        Same items, reordered in a U-shaped attention pattern.
    """
    if len(items) <= 2:
        return items  # nothing to reorder

    # Sort by relevance descending (most important first)
    ranked = sorted(
        items,
        key=lambda x: x.get(key_field, 0.0),
        reverse=True,
    )

    # Build U-shaped order: best items at start and end
    result: List[Dict] = [{}] * len(ranked)
    front = 0
    back = len(ranked) - 1

    for i, item in enumerate(ranked):
        if i % 2 == 0:
            result[front] = item
            front += 1
        else:
            result[back] = item
            back -= 1

    return result
