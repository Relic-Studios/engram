# Research Findings: State-of-the-Art Improvements for Engram Memory Retrieval and Code Generation

**Research executed**: 2026-02-18
**Source prompt**: `docs/research_prompts/sota-memory-retrieval-and-generation.md`
**Supplementary data**: `Benchmarking AI Memory Systems_ Engram.md` (335-line analysis, 70+ citations)

---

## Executive Summary: Top 10 Improvements by Impact

| # | Improvement | Impact | Effort | Latency Cost | Priority |
|---|---|---|---|---|---|
| 1 | Replace 50/50 merge with Reciprocal Rank Fusion (RRF) | +15-25% precision | ~2 hours | ~0ms | **Immediate** |
| 2 | Add FTS5 Porter stemming | +10-15% recall | ~15 min | ~0ms | **Immediate** |
| 3 | Upgrade embedding model (all-MiniLM-L6-v2 -> nomic-embed-text-v1.5) | +20-30% retrieval quality | ~4 hours | +5ms/query | **This week** |
| 4 | Add cross-encoder reranking after hybrid search | +10-15% precision@5 | ~4 hours | +30-50ms | **This week** |
| 5 | Dynamic context allocation (replace fixed shares) | +variable quality | ~8 hours | ~0ms | **This week** |
| 6 | Fix archived message bug in get_recent_messages | Fix correctness | ~30 min | ~0ms | **Immediate** |
| 7 | Add query expansion (person-aware + lightweight) | +5-10% recall | ~4 hours | +10-20ms | **Next sprint** |
| 8 | Topic-coherent consolidation (BERTopic clustering) | Better thread quality | ~16 hours | N/A (offline) | **Next sprint** |
| 9 | Code-specific trace metadata and retrieval profile | Better code-gen context | ~8 hours | ~0ms | **Next sprint** |
| 10 | LongLLMLingua-style prompt compression | 2-4x token savings | ~16 hours | +20-40ms | **Next month** |

---

## Section 0: Hybrid Search Architecture

### Finding 0a: Reciprocal Rank Fusion (RRF) crushes naive weighted averaging

**Current state**: `engram/search/unified.py` uses `combined_score = 0.5 * fts_score + 0.5 * sem_score` with asymmetric normalization (local min-max for FTS, global /2.0 for cosine).

**The problem**: This approach has three flaws:
1. FTS min-max normalization is query-local, making scores incomparable across queries
2. The 50/50 weight is arbitrary and not tunable
3. Items appearing in only one source get a default 0.5, biasing hybrid results

**RRF formula**: `score(d) = sum(1 / (k + rank_i(d)))` for each retrieval system `i`, where `k=60` is standard.

**Why RRF is better**:
- Uses **ranks** not scores, eliminating the normalization asymmetry problem entirely
- Proven in TREC/CLEF evaluations to outperform linear combination by 10-25% on nDCG
- Used by Elasticsearch, Weaviate, and Vespa as their default hybrid merge strategy
- Zero-parameter (k=60 is robust across datasets) or single-parameter (tunable k)
- Items appearing in only one system get naturally lower combined scores (no artificial 0.5 default)

**Implementation in engram**:
```python
def rrf_merge(fts_results, sem_results, k=60):
    scores = {}
    for rank, doc in enumerate(fts_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(sem_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Recommendation**: Replace `_merge_results()` in `unified.py` with RRF. This is a ~50-line change. **Highest ROI improvement in the entire codebase.**

---

### Finding 0b: Cross-encoder reranking is the single biggest precision booster

**Current state**: No reranking. Hybrid results go directly to the knapsack allocator.

**Research consensus** (Wang et al. 2024 "Searching for Best Practices in RAG"):
- Adding a cross-encoder reranker after initial retrieval is the **#1 most impactful RAG optimization**
- Typical precision@5 improvement: +10-18% over no-reranking baseline
- The optimal pipeline is: Retrieve ~50 candidates -> Rerank to top 10 -> Pack into context

**Best local rerankers for RTX 4090**:

| Model | Size | Latency (30 docs) | BEIR nDCG | Notes |
|---|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M | ~30ms | 0.43 | Best speed/quality tradeoff |
| `BAAI/bge-reranker-v2-m3` | 568M | ~120ms | 0.49 | Best quality, fits in 4090 VRAM |
| `jinaai/jina-reranker-v2-base-multilingual` | 278M | ~60ms | 0.47 | Good multilingual support |
| FlashRank (ONNX optimized) | 22M | ~10ms | 0.40 | Fastest, minimal quality |

**Recommendation**: Start with `ms-marco-MiniLM-L-12-v2` (30ms for 30 candidates). If quality matters more than latency, upgrade to `bge-reranker-v2-m3`. Both run comfortably on RTX 4090.

**Integration point**: Add reranking step between `unified.search()` and the knapsack allocator in `pipeline/before.py`. Retrieve 50 candidates, rerank, take top 10-15.

---

### Finding 0c: Query expansion provides diminishing but real returns

**HyDE (Hypothetical Document Embedding)**: Generate a hypothetical answer, embed it, search with that embedding instead of the raw query.
- Improves recall by 5-15% on knowledge-intensive queries
- **Problem for engram**: Requires an LLM call (+200-500ms), which breaks the sub-200ms before-pipeline target
- **Verdict**: Not viable for real-time pipeline. Could be used for background reindexing or offline tasks.

**Person-aware expansion**: Engram knows relationships. When a query mentions "the coffee shop," expansion to "Hel's coffee shop" using the relationship store is cheap (string lookup, no LLM).
- Low-latency (~1ms)
- High-value for engram's specific use case (personal memory)
- **Recommendation**: Implement person-aware entity expansion as a pre-search step in `before.py`

**Multi-query retrieval**: Decompose a complex query into sub-queries, retrieve for each, merge.
- Overkill for engram's typical queries (short conversational messages)
- **Verdict**: Skip for now.

---

### Finding 0d: Embedding model upgrade is critical

**Current state**: ChromaDB defaults to `all-MiniLM-L6-v2` (384d, 22M params, MTEB average ~0.56). This model is from 2022 and has been significantly surpassed.

**Top local embedding models (2025-2026 MTEB rankings)**:

| Model | Dims | Size | MTEB Avg | RTX 4090 Speed | Notes |
|---|---|---|---|---|---|
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 137M | 0.63 | ~5ms/doc | **Best for engram** - Matryoshka, local, open |
| `BAAI/bge-large-en-v1.5` | 1024 | 335M | 0.64 | ~8ms/doc | Slightly better quality, larger |
| `intfloat/e5-large-v2` | 1024 | 335M | 0.62 | ~8ms/doc | Instruction-tuned, good for queries |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 335M | 0.65 | ~8ms/doc | Top performer, Matryoshka support |
| `all-MiniLM-L6-v2` (current) | 384 | 22M | 0.56 | ~2ms/doc | **Outdated** |

**Recommendation**: Switch to `nomic-embed-text-v1.5`:
- +7 points on MTEB average (0.56 -> 0.63) = ~12% better retrieval
- Matryoshka support: can use 256d for fast initial search, 768d for reranking
- Apache 2.0 license, runs locally, no API dependency
- Reasonable size (137M fits easily in 4090 VRAM alongside the LLM)

**Migration path**: 
1. Add `embedding_model` to `Config` dataclass
2. Build embedding function using `sentence-transformers` 
3. Pass to `SemanticSearch.__init__()` 
4. Run `reindex_all()` once (one-time cost, ~30 min for 50k traces on 4090)

---

## Section 1: FTS5 Improvements

### Finding 1a: Porter stemming is a free win

**Current state**: FTS5 uses default `unicode61` tokenizer. No stemming, no synonyms.

**Impact**: "running" does not match "run". "memories" does not match "memory". This silently drops valid results.

**Fix** (in `episodic/store.py` schema creation):
```sql
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts 
USING fts5(content, content=messages, content_rowid=rowid, tokenize='porter unicode61');
```

**Caveat**: Changing the tokenizer requires rebuilding FTS indexes. Add a migration step or do it during reindexing.

**Recommendation**: Implement alongside the embedding model migration (both require reindexing anyway).

### Finding 1b: BM25 tuning for short texts

**Default BM25**: k1=1.2, b=0.75. Tuned for 200-500 word documents.

**Engram's traces**: Typically 50-200 tokens (much shorter than BM25 defaults expect).

**Research finding**: For short texts, reducing `b` (length normalization) prevents over-penalizing short documents. Recommended for engram: k1=1.5, b=0.3-0.5.

**Problem**: SQLite FTS5 does NOT expose BM25 parameter tuning. The rank function is hardcoded.

**Workaround options**:
1. Use a custom ranking function via `fts5_custom_rank()` — complex, fragile
2. Accept default BM25 and rely on reranking to correct rank errors — **recommended**
3. Replace FTS5 with SPLADE sparse vectors — high effort, high reward, future consideration

**Recommendation**: Accept default BM25 for now. The reranker will fix most ranking issues. Revisit SPLADE when the retrieval pipeline is more mature.

---

## Section 2: Context Assembly Optimization

### Finding 2a: Dynamic allocation dramatically outperforms fixed shares

**Current state**: Identity always gets 16% of tokens regardless of whether the query needs identity context. A factual recall query wastes 960 tokens on SOUL.md.

**Research finding** (multiple RAG best-practices papers): Query-dependent allocation improves downstream response quality by 15-30% over fixed allocation.

**Lightweight query classification** (no LLM required):
```python
PROFILES = {
    "identity":    {"identity": 0.25, "relationship": 0.20, "episodic": 0.10, ...},
    "factual":     {"identity": 0.05, "relationship": 0.05, "episodic": 0.40, ...},
    "emotional":   {"identity": 0.15, "relationship": 0.25, "episodic": 0.20, ...},
    "code":        {"identity": 0.03, "procedural": 0.35, "episodic": 0.25, ...},
    "default":     {"identity": 0.16, ...},  # current allocation
}
```
Classify using keyword matching: if message contains code keywords (function, debug, error, implement, build) -> "code" profile. If emotional keywords -> "emotional" profile. Else "default".

**"Lost in the Middle" optimization** (Liu et al. 2023): LLMs attend more to the start and end of the context window. Place the most relevant episodic traces at the **start**, identity at the **end** (as grounding), and lower-priority content in the middle.

**Recommendation**: Implement query-type detection (keyword-based, ~50 lines) and context profiles. Reorder sections to exploit "lost in the middle" effect.

### Finding 2b: LongLLMLingua-style compression for token savings

**LongLLMLingua** (Jiang et al., ACL 2024): Prompt compression that achieves **21.4% performance boost with 4x fewer tokens** on NaturalQuestions.

**How it works**: Uses a small local LLM to score each token's importance via perplexity, then drops low-importance tokens while preserving key information.

**For engram**: Could compress the identity/relationship sections from ~1000 tokens to ~250 tokens with minimal information loss. The saved tokens go to episodic traces (the most query-relevant content).

**Latency cost**: ~20-40ms on RTX 4090 using a small model (e.g., Qwen2.5-0.5B).

**Recommendation**: Medium-term improvement. Implement after the higher-priority items (RRF, reranking, embedding upgrade).

### Finding 2c: Knapsack should use MMR for diversity

**Current state**: Greedy 0/1 knapsack by salience/token density. Can select multiple near-duplicate traces.

**Maximal Marginal Relevance (MMR)**: `score(d) = lambda * relevance(d) - (1-lambda) * max_sim(d, selected)`. Standard approach for diversifying retrieval results.

**Recommended lambda**: 0.7 (70% relevance, 30% diversity) for memory traces.

**Implementation**: After reranking, apply MMR during knapsack selection. Compare each candidate's cosine similarity to already-selected items and penalize near-duplicates.

---

## Section 3: Vector Search Architecture

### Finding 3a: ColBERT is overkill for engram's scale

ColBERT stores per-token embeddings (384 floats x ~100 tokens per trace = ~153KB per trace). At 50k traces, this is ~7.5GB of index — feasible on RTX 4090 but excessive.

**Verdict**: Not worth the complexity for <100k traces. Bi-encoder + cross-encoder reranker achieves 95% of ColBERT's quality with much simpler infrastructure.

### Finding 3b: BGE-M3 is interesting but premature

BGE-M3 produces dense, sparse, AND ColBERT representations in a single forward pass. This could replace both FTS5 and ChromaDB with a single model.

**Problem**: Requires replacing the entire search pipeline. High risk, high effort.

**Recommendation**: Monitor BGE-M3 adoption. When engram reaches v1.0, evaluate as a unification strategy.

---

## Section 4: Consolidation Improvements

### Finding 4a: Topic-coherent clustering is the biggest consolidation improvement

**Current state**: Temporal-only clustering (72h window). Two conversations about different topics 1 hour apart get merged.

**BERTopic approach**: Use the embedding model to cluster traces by semantic similarity, then apply temporal constraints within each topic cluster.

**Algorithm**:
1. Embed all unconsolidated traces (already in ChromaDB)
2. Cluster with HDBSCAN (density-based, handles noise, no need to specify k)
3. Within each cluster, apply temporal proximity splitting (current 72h window)
4. Only consolidate clusters with 5+ traces (current threshold)

**Latency**: Offline process, no impact on query latency. HDBSCAN over 200 traces takes <1s on CPU.

**Dependencies**: `hdbscan` package (pip install), embeddings already computed.

**Recommendation**: Implement as a replacement for the temporal-only clustering in `consolidator.py`. High value for thread/arc quality.

### Finding 4b: Faithfulness verification via NLI

**Problem**: LLM consolidation can hallucinate facts not in source traces.

**Solution**: After generating a thread summary, run each claim through an NLI model (e.g., `facebook/bart-large-mnli` or `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) to verify entailment from source traces.

**Practical approach**: Score = average entailment probability across claims. If score < 0.7, fall back to extractive summary.

**Latency**: ~100ms per summary on RTX 4090. Acceptable for offline consolidation.

**Recommendation**: Add as a post-consolidation verification step. Logs faithfulness scores for monitoring.

---

## Section 5: Knowledge Graph vs YAML

### Finding: Lightweight graph edges, not a full graph database

Zep's Graphiti uses Neo4j — massive overhead for engram's scale (hundreds of entities, not millions).

**Recommendation**: Add explicit relationship edges to the existing SQLite database as a new `relationships` table:
```sql
CREATE TABLE relationships (
    id TEXT PRIMARY KEY,
    subject TEXT,       -- entity name
    predicate TEXT,     -- relationship type
    object TEXT,        -- entity name
    valid_from TEXT,    -- ISO timestamp
    valid_until TEXT,   -- ISO timestamp (NULL = still valid)
    confidence REAL,
    source_trace_id TEXT,
    metadata TEXT       -- JSON
);
```

This gives temporal validity (like Zep) and relationship traversal (1-hop graph queries via SQL JOINs) without adding Neo4j as a dependency. Simple enough to implement in ~200 lines of Python.

---

## Section 6: Code Generation Memory

### Finding 6a: Code-specific embedding models exist but may not help

Voyage Code, CodeBERT, and StarCoder embeddings are trained on code. But engram's code memories aren't raw code — they're natural language descriptions of coding patterns, decisions, and debugging strategies.

**Recommendation**: Stick with the general-purpose `nomic-embed-text-v1.5` for all traces. If code retrieval quality is poor, consider a second ChromaDB collection with a code-specific model.

### Finding 6b: Structured metadata on code traces is high-value

When traces describe code activities, extract and store structured metadata:
- `language`: python, typescript, sql, etc.
- `activity`: debugging, implementation, refactoring, architecture
- `files_mentioned`: extracted file paths
- `error_types`: extracted error patterns

This enables filtered retrieval: "when debugging Python before" -> filter `language=python, activity=debugging`.

**Implementation**: Add metadata extraction rules to `signal/extract.py` (keyword-based, not LLM) for code-related exchanges.

### Finding 6c: Code context profile for the allocator

When the query-type classifier detects a "code" query, switch to:
- Procedural skills: 35% (up from 6%)
- Episodic traces: 25% (up from 16%, filtered to code-related)
- Recent conversation: 20%
- Identity: 3% (minimal — code tasks don't need SOUL.md)
- Relationship: 2%
- Reserve: 15%

---

## Section 7: Reinforcement and Decay

### Finding 7a: Citation-only reinforcement is better than blanket reinforcement

**Current state**: ALL context traces get +0.05 when signal > 0.7. A trace that was irrelevant but happened to be loaded gets the same boost as one that was central to the response.

**Research finding**: The citation bonus mechanism (already implemented at +0.025 for `[N]` cited traces) is more principled.

**Recommendation**: Make citation-based reinforcement the **primary** mechanism:
- Cited traces: +0.05 (full reinforce_delta)
- Uncited context traces: +0.01 (minimal "you were present" signal)
- Low-signal exchanges: -0.03 for all context traces (keep current)

This prevents "salience inflation" where everything drifts to 1.0 over time.

### Finding 7b: Access recency should modulate decay resistance

**Current state**: `resistance = 1 / (1 + access_count * 0.1)` — no recency weighting.

**Fix**: `resistance = 1 / (1 + recent_access_count * 0.1)` where `recent_access_count` counts accesses in the last 30 days. Old accesses fade from the resistance calculation.

**Implementation**: Add `recent_accessed_since` parameter to `decay_pass()` that only counts accesses after `now - 30 days`.

---

## Section 8: Local Infrastructure Optimization

### Finding 8a: Parallelize the after-pipeline LLM calls

**Current state**: Signal measurement (~400ms) and semantic extraction (~600ms) run sequentially.

**Fix**: Run them concurrently with `asyncio.gather()` or `concurrent.futures.ThreadPoolExecutor`. Total after-pipeline: ~600ms instead of ~1200ms.

### Finding 8b: Batch SQLite writes

**Current state**: Each `log_message()`, `log_trace()`, `reinforce()` does its own `conn.commit()`.

**Fix**: Accumulate writes in a batch and commit once at the end of the after-pipeline. Under WAL mode, this reduces disk I/O by 5-10x.

### Finding 8c: Pre-compute embeddings at write time (already done)

Engram already indexes traces into ChromaDB on write via `_on_trace_logged`. This is correct. No change needed.

---

## Section 9: Cognitive Architecture Alignment

### Finding 9a: ACT-R's base-level activation validates engram's approach (with corrections)

ACT-R's memory activation formula: `B_i = ln(sum(t_j^-d))` where `t_j` are times since each access and `d=0.5`.

**Key difference**: ACT-R counts recency AND frequency of access, with recent accesses weighted more heavily. Engram's access_count has no recency weighting.

**Recommendation**: Adopt ACT-R-style recency weighting (finding 7b above).

### Finding 9b: Complementary Learning Systems theory supports episodic->semantic transfer

CLS theory: hippocampus (episodic) rapidly encodes specific experiences; neocortex (semantic) slowly extracts patterns. This maps to engram's episode->thread->arc pipeline.

**CLS implication**: The "sleep consolidation" phase (replay and re-evaluate during idle time) would improve engram's consolidation quality. Engram currently only consolidates reactively.

**Recommendation**: Add an idle-time consolidation mode (in `runtime/modes.py` QUIET_PRESENCE or SLEEP modes) that proactively consolidates, reindexes, and runs decay.

---

## Implementation Roadmap

### Phase 1: Immediate Wins (1-2 days, config changes + small code edits)

1. **Replace 50/50 merge with RRF** in `unified.py` (~50 lines)
2. **Add Porter stemming** to FTS5 table creation in `store.py` (~2 lines + reindex)
3. **Fix archived message bug** in `get_recent_messages()` (~5 lines)
4. **Fix update_access() no-op** for messages in `store.py` (~10 lines)
5. **Add embedding_model to Config** dataclass (~20 lines)

### Phase 2: High-Impact Upgrades (1 week)

6. **Switch to nomic-embed-text-v1.5** + reindex all collections
7. **Add cross-encoder reranker** (ms-marco-MiniLM-L-12-v2) in before-pipeline
8. **Implement query-type classification** (keyword-based) + dynamic context profiles
9. **Reorder context sections** for "lost in the middle" optimization
10. **Make citation-based reinforcement primary** (adjust `reinforcement.py`)

### Phase 3: Structural Improvements (2-3 weeks)

11. **Topic-coherent consolidation** with HDBSCAN clustering
12. **NLI faithfulness verification** for consolidation summaries
13. **Code-specific trace metadata** extraction and retrieval profile
14. **Lightweight relationship graph** in SQLite
15. **Parallelize after-pipeline** LLM calls
16. **Batch SQLite writes** in after-pipeline

### Phase 4: Advanced (1-2 months)

17. **LongLLMLingua prompt compression** for identity/relationship sections
18. **MMR diversity** in knapsack allocator
19. **Person-aware query expansion** in before-pipeline
20. **Idle-time proactive consolidation** (CLS-inspired)
21. **Access recency** in decay resistance formula
22. **BGE-M3 evaluation** for unified sparse+dense retrieval

---

## Dependencies to Add

```toml
[project.optional-dependencies]
retrieval = [
    "sentence-transformers>=2.3.0",  # embedding model + reranker
    "hdbscan>=0.8.33",               # topic clustering
]
```

Note: `sentence-transformers` pulls in `torch`, which is already needed for the embedding model. On RTX 4090 with CUDA, all models run with GPU acceleration automatically.

---

## Expected Outcomes

If Phase 1-2 are implemented:
- **Retrieval precision@5**: +25-40% over current (RRF + reranker + better embeddings)
- **Retrieval recall**: +15-20% (stemming + better embeddings)
- **Context relevance**: +15-30% (dynamic allocation + reordering)
- **Before-pipeline latency**: <200ms at 10k traces (within Zep's target)
- **LoCoMo competitive**: Should approach Letta's 74% baseline

If Phase 3-4 are also implemented:
- **Consolidation quality**: Topic-coherent threads with faithfulness verification
- **Code generation**: Purpose-built context profiles for coding tasks
- **Identity coherence**: Maintained through improved signal + citation-only reinforcement
- **Publication-ready**: Ablation studies possible by toggling each component
