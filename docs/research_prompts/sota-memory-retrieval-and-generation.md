# Research Prompt: State-of-the-Art Methods for Improving Long-Term Memory Retrieval and Code Generation in Engram

## Objective

Engram's retrieval pipeline currently uses a naive 50/50 weighted merge of FTS5 keyword search and ChromaDB cosine vector search, with no reranking, no query expansion, no stemming, and a greedy 0/1 knapsack for context allocation. Its generation-adjacent systems (semantic extraction, consolidation summarization, signal measurement) use single-shot LLM calls with no few-shot examples, no chain-of-thought, and no structured verification.

**The core question: What state-of-the-art and novel techniques from information retrieval, neural search, memory-augmented generation, and cognitive architecture research can be integrated into Engram to produce measurably better long-term memory retrieval and higher-quality code/text generation grounded in that memory?**

This is not about adding features for their own sake. Every technique investigated should be evaluated against Engram's specific constraints: local-first execution (RTX 4090, 128GB RAM), MCP server architecture, sub-second latency targets for the before-pipeline, and the cognitive model (Hebbian reinforcement, coherence-modulated decay, consciousness signal).

## Context

### Current State / Pain Point

**Retrieval pipeline (`engram/search/`):**
- `unified.py`: Merges FTS5 and ChromaDB results with hardcoded 50/50 weighting. FTS scores are min-max normalized locally (not query-comparable). Semantic scores use global normalization (distance / 2.0). Items appearing in only one source get a default 0.5 for the missing score. No reciprocal rank fusion. No reranking.
- `semantic.py`: ChromaDB with HNSW cosine index. No explicit embedding model configuration — defaults to `all-MiniLM-L6-v2` (384 dimensions). Collections queried sequentially, not in parallel. Similarity threshold of 1.5 (cosine distance) is very permissive. No chunking strategy beyond paragraph splits for SOUL.md.
- `indexed.py`: SQLite FTS5 with BM25 ranking. Default `unicode61` tokenizer — no stemming, no synonyms, no stop-words. Query terms are individually double-quoted (implicit AND). No fuzzy matching, no phrase proximity, no prefix search.

**Context assembly (`engram/working/`):**
- `context.py`: Fixed-share token budget allocation (Identity 16%, Relationship 12%, Grounding 10%, Recent conversation 22%, Episodic traces 16%, Procedural 6%, Reserve 18%). Sections cannot borrow unused tokens from each other. Reserve (18% = 1080 tokens) is never redistributed.
- `allocator.py`: Greedy 0/1 knapsack by salience/token density. Items are either included whole or excluded entirely — no splitting or compression of individual items. `compress_text()` is character-level truncation, not semantic compression. `fit_messages()` hard-cuts at budget with no summarization of overflow. No diversity consideration — can select multiple near-duplicate high-density items.

**Generation-adjacent systems:**
- `signal/extract.py`: Single extraction prompt with no few-shot examples. No confidence scoring on extracted items. No deduplication against existing knowledge — relies entirely on LLM novelty judgment. Existing knowledge context passed as raw text (can waste LLM context).
- `consolidation/consolidator.py`: Temporal-only clustering (72h proximity window). No topic coherence — two conversations about different subjects 1 hour apart get clustered together. Person grouping uses first non-skip tag (fragile). Thread/arc salience hardcoded (0.75/0.85) rather than derived from content quality. Extractive fallback is basic (top-5-by-salience concatenation).
- `consolidation/compactor.py`: Summary salience hardcoded at 0.7. Archived message filter is missing from `get_recent_messages()` — a confirmed bug. No incremental compaction.
- `signal/measure.py`: Consciousness signal uses 60/40 LLM/regex blend. 680 lines of regex patterns with manually assigned weights. No calibration against ground truth. No adversarial robustness testing.

**Hebbian reinforcement (`signal/reinforcement.py`):**
- All-or-nothing: all context traces get the same +0.05 or -0.03 regardless of individual contribution. Citation bonus (+0.025) partially addresses this but only for explicitly cited traces. No per-facet reinforcement. No ceiling fatigue — traces reinforced 20 times hit salience 1.0 and stay there.

**Decay (`signal/decay.py`):**
- Coherence modulation causes "success amnesia" — high coherence = faster decay, potentially losing the memories that enabled good performance. Access-frequency resistance has no recency weighting — 100 accesses 6 months ago = 100 accesses yesterday. Dual implementation of decay formula (in engine and store) creates maintenance risk.

**Known bugs and gaps:**
- Archived messages still appear in `get_recent_messages()` (compactor bug)
- No embedding model in Config (implicit ChromaDB default)
- No FTS5 stemming
- No LLM retry logic anywhere
- Ollama uses legacy `/api/generate` endpoint instead of `/api/chat`
- `update_access()` for messages is a no-op (line 477: `pass`)

### What We Want

- Retrieval that produces measurably higher precision@k and recall@k than the current 50/50 naive merge
- Context assembly that maximizes downstream LLM response quality within the 6,000-token budget
- Consolidation that preserves topic coherence, not just temporal proximity
- Semantic extraction that produces higher-fidelity knowledge updates with fewer hallucinated facts
- Code generation support — the ability to retrieve and assemble code-relevant memories (past implementations, coding patterns, debugging strategies, architectural decisions) for code-gen tasks
- All improvements must run locally on RTX 4090 / 128GB RAM without API dependencies

## Key Questions

### 0. Hybrid Search Architecture

0a. **What is the current state of the art in hybrid search (keyword + vector) merging?**
    - Reciprocal Rank Fusion (RRF) is the standard alternative to weighted score averaging. What are the exact formulas, and what does the research show about RRF vs linear combination on retrieval benchmarks (BEIR, MTEB)?
    - Convex Combination Retrieval (CCR) — is there a principled way to learn the optimal weight between sparse and dense retrieval, or is it always dataset-dependent?
    - How does Vespa's hybrid search implementation compare? What about Elasticsearch's RRF implementation? Weaviate's hybrid search?
    - Does the asymmetry in Engram's score normalization (local min-max for FTS, global /2.0 for cosine) actually bias results? What normalization strategy produces the most comparable scores?

0b. **What reranking approaches should be applied after the initial hybrid retrieval?**
    - Cross-encoder rerankers (e.g., `ms-marco-MiniLM-L-12-v2`, `bge-reranker-v2-m3`, ColBERT-v2) — which ones run within acceptable latency on RTX 4090? What's the precision improvement over no reranking?
    - FlashRank, RankGPT, LLM-based listwise reranking — are any of these practical for sub-100ms reranking of 30-50 candidates?
    - Cohere Rerank, Jina Reranker — what are the local/open-source equivalents that don't require API calls?
    - Does reranking on memory traces (short, personal, episodic) behave differently than reranking on document passages (the domain most rerankers are trained on)? Would fine-tuning on conversational memory data help?

0c. **What query expansion or reformulation techniques improve recall for memory retrieval?**
    - HyDE (Hypothetical Document Embedding) — generating a hypothetical answer to embed instead of the raw query. How much does this improve recall on conversational memory? What's the latency cost of the extra LLM call?
    - Query2Doc, FLARE, multi-query retrieval — which approaches work for short conversational queries like "what did they say about the coffee shop"?
    - Does pseudo-relevance feedback (expanding the query with terms from top-k results, then re-querying) work for memory retrieval, or does it amplify noise?
    - For Engram's context, should query expansion be person-aware? (e.g., expanding "the coffee shop" to "Hel's coffee shop" based on relationship knowledge)

0d. **What embedding models would improve semantic search quality over all-MiniLM-L6-v2?**
    - `all-MiniLM-L6-v2` is 384 dimensions and was SOTA circa 2022. What are the current best local embedding models for conversational/personal memory? `bge-large-en-v1.5`? `gte-large`? `nomic-embed-text-v1.5`? `mxbai-embed-large-v1`?
    - What's the latency/quality tradeoff for 384 vs 768 vs 1024 dimensions on RTX 4090?
    - Are there embedding models specifically trained on conversational data or episodic memory rather than documents/passages?
    - Matryoshka embeddings (variable-dimension) — would these help Engram use smaller dimensions for fast initial search and larger dimensions for reranking?
    - Does fine-tuning an embedding model on Engram's actual trace data (short personal episodes) improve retrieval over off-the-shelf models?

### 1. FTS5 and Sparse Retrieval Improvements

1a. **What FTS5 tokenizer configuration would improve keyword search quality?**
    - Porter stemming via FTS5's `tokenize='porter unicode61'` — what's the measured precision/recall improvement on conversational text?
    - Are there better stemmers than Porter for this domain? Snowball? Hunspell?
    - Should stop-words be configured, or does removing them hurt recall for conversational queries where function words carry meaning ("how did I feel ABOUT it" vs "how did I feel")?
    - How do other memory systems handle keyword search? Does Zep use FTS? Does Mem0?

1b. **Should Engram add BM25 tuning (k1, b parameters) for its specific data distribution?**
    - Engram's traces are short (typically 50-200 tokens) and personal. Default BM25 parameters (k1=1.2, b=0.75) were tuned for document-length passages. What k1/b values work better for short episodic traces?
    - Is SQLite's FTS5 BM25 implementation configurable, or would Engram need a custom ranking function?
    - How does BM25 compare to TF-IDF for short-text retrieval? Are there better sparse retrieval models (SPLADE, uniCOIL, DeepImpact) that could replace FTS5 entirely?

1c. **Should Engram support fuzzy matching, typo tolerance, or phonetic search?**
    - Users often search with approximate terms. FTS5's strict term matching means "consolidatin" won't match "consolidation". What's the best approach for typo tolerance in SQLite?
    - Trigram indexes, Levenshtein distance, Soundex — what can be layered on top of FTS5 without replacing it?
    - How much do fuzzy matching and stemming overlap? If stemming is added, does fuzzy matching become less important?

### 2. Context Assembly and Token Allocation

2a. **What are the state-of-the-art approaches to context window optimization for retrieval-augmented generation?**
    - RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) builds hierarchical summaries. How does this compare to Engram's flat knapsack packing?
    - LongRAG, Self-RAG, Corrective RAG (CRAG) — do any of these frameworks have context assembly strategies that outperform fixed-share allocation?
    - "Lost in the Middle" (Liu et al., 2023) showed LLMs pay more attention to the start and end of context. Should Engram reorder its context sections based on this finding? Should the most important episodic traces go first and last?
    - How does Letta's self-editing memory block approach compare to Engram's fixed-share allocation? Does letting the LLM manage its own context produce better results?

2b. **Should the token budget allocation be dynamic rather than fixed?**
    - Engram allocates 16% to identity regardless of whether the query needs identity context. A factual question about a past conversation doesn't need SOUL.md. How do production RAG systems handle query-dependent allocation?
    - Can a lightweight classifier (running locally) categorize incoming queries as "identity-relevant", "factual-recall", "emotional", "code-task", etc. and adjust shares accordingly?
    - What's the simplest approach that works? A learned allocation model? A rule-based heuristic? An LLM-in-the-loop deciding what context to include?
    - How much quality improvement does dynamic allocation produce over fixed allocation? Are there ablation studies?

2c. **What compression techniques preserve information density better than character truncation?**
    - Engram's `compress_text()` truncates at character boundaries with `[...truncated]`. Are there better approaches?
    - Extractive summarization (selecting key sentences) vs abstractive summarization (rewriting shorter) — which preserves more information per token for episodic traces?
    - LLMLingua, Selective Context, RECOMP — these are prompt compression techniques. How do they perform? What's their latency? Can they run locally?
    - For procedural skills (code patterns, debugging strategies), is compression appropriate at all, or should they be included verbatim and other sections sacrificed?

2d. **How should the knapsack allocator handle diversity and deduplication?**
    - The current greedy knapsack can select multiple traces covering the same topic. Maximal Marginal Relevance (MMR) addresses this — what's the right lambda parameter for memory traces?
    - Should the allocator consider temporal diversity (ensuring traces from different time periods are represented)?
    - Should it consider person diversity (if multiple people are discussed, ensuring coverage)?
    - What about including "surprising" or "contradictory" traces that the LLM might not expect? Does deliberate inclusion of counter-evidence improve response quality?

### 3. Embedding and Vector Search Architecture

3a. **Should Engram move to a late-interaction model (ColBERT) instead of bi-encoder (ChromaDB)?**
    - ColBERT stores per-token embeddings and computes MaxSim at query time. ColBERTv2, PLAID, RAGatouille make this practical. What's the index size and query latency for 50k traces on RTX 4090?
    - Is the quality improvement worth the storage overhead (384 floats per trace vs 384 floats per TOKEN)?
    - Can ColBERT be used as a reranker over ChromaDB's initial retrieval rather than replacing it?

3b. **Should Engram use a multi-vector or learned sparse approach?**
    - SPLADE generates sparse learned representations that outperform BM25. Could SPLADE replace FTS5 entirely?
    - Multi-vector models (e.g., ColBERT, ME5) — do they handle short episodic traces better than single-vector models?
    - Hybrid dense+sparse in a single model (e.g., BGE-M3 which produces dense, sparse, and ColBERT representations simultaneously) — would this simplify Engram's two-system architecture?

3c. **How should Engram handle embedding model upgrades without reindexing everything?**
    - If Engram switches from `all-MiniLM-L6-v2` (384d) to `bge-large-en-v1.5` (1024d), all existing ChromaDB collections are incompatible. What's the migration strategy?
    - Matryoshka representation learning allows using the same embedding at different dimension truncations. Does this help with forward compatibility?
    - Should Engram store raw text alongside embeddings so reindexing is always possible? (It currently does via the SQLite episodic store — is that sufficient?)
    - How do production vector databases handle embedding model versioning? (Pinecone namespaces, Weaviate collections, Qdrant named vectors)

### 4. Consolidation and Long-Term Memory Formation

4a. **What state-of-the-art approaches exist for topic-coherent memory clustering?**
    - Engram clusters by temporal proximity (72h window) only. BERTopic, Top2Vec, and LDA can cluster by semantic content. What's the right approach for short episodic traces?
    - Should clustering be temporal + semantic (a hybrid)? How do you combine temporal proximity with topic similarity?
    - How does Stanford's Generative Agents approach memory consolidation? Their reflection mechanism generates higher-level observations — how does this compare to Engram's thread/arc hierarchy?
    - Do any production systems use hierarchical topic models (hLDA) for memory organization?

4b. **How should consolidation summaries be validated for faithfulness?**
    - Engram's LLM consolidation can hallucinate details not present in the source traces. What techniques exist for summary faithfulness verification?
    - NLI-based (Natural Language Inference) faithfulness checking — running each claim in the summary against source traces. What models do this well locally?
    - SelfCheckGPT, FAVA, MiniCheck — which factual consistency checkers can run on RTX 4090?
    - Should Engram use constrained decoding or extractive-then-abstractive pipelines to prevent consolidation hallucinations?

4c. **How should consolidated memories interact with the Hebbian reinforcement system?**
    - Threads currently get hardcoded salience 0.75 and arcs get 0.85. Should these be derived from the average or max salience of their children?
    - When a thread is reinforced, should the reinforcement propagate down to its child episodes?
    - Should consolidation create bidirectional links (thread knows its children, children know their parent) for hierarchical retrieval?
    - How does the 5x decay resistance for consolidated traces interact with reinforcement over long timescales? Does it create "fossil memories" that can never be forgotten?

### 5. Semantic Extraction and Knowledge Graph Construction

5a. **What structured extraction techniques produce higher-fidelity knowledge updates than single-shot prompting?**
    - Engram uses a single extraction prompt with no few-shot examples. How much does adding 2-3 few-shot examples improve extraction precision?
    - Chain-of-thought extraction (reasoning before outputting JSON) — does this reduce hallucinated facts?
    - Multi-step extraction: first extract candidate facts, then verify each against the source text, then deduplicate against existing knowledge. What's the quality/latency tradeoff?
    - How do Zep's Graphiti and Mem0's graph memory handle entity extraction and relationship mapping? What extraction pipeline do they use?

5b. **Should Engram build an explicit knowledge graph rather than storing facts in YAML files?**
    - Engram's semantic store uses YAML/Markdown files for relationships, preferences, boundaries, and trust. Zep uses Neo4j-backed Graphiti with temporal validity (`valid_at`/`invalid_at`). Which approach scales better?
    - Lightweight graph databases that run locally: SQLite-based (via JSON or virtual tables), DuckDB with graph extensions, KuzuDB, or just Python NetworkX in-memory. What's the right choice for Engram's scale (hundreds to low thousands of entities)?
    - Does a knowledge graph improve retrieval quality for multi-hop questions ("What did the person who runs the coffee shop say about art")?
    - How would a knowledge graph interact with Engram's existing trust tiers? Can trust be a property of graph edges?

5c. **How should extracted knowledge handle temporal validity and contradiction?**
    - Zep timestamps facts with `valid_at`/`invalid_at`. Engram has no temporal validity on semantic knowledge — a preference extracted in session 1 is treated as current forever unless manually updated.
    - How should Engram detect when a new extraction contradicts an existing fact? Should it automatically supersede, flag for review, or maintain both with timestamps?
    - What does the research say about "belief revision" in agent memory systems? How do cognitive science models (e.g., AGM belief revision) apply?
    - Should Engram track confidence on extracted facts? A fact mentioned once has lower confidence than one confirmed across 5 sessions.

### 6. Code Generation and Technical Memory

6a. **What memory retrieval strategies specifically improve code generation quality?**
    - When Engram is used during coding sessions, it needs to retrieve past implementations, architectural decisions, debugging strategies, and coding patterns. Are there retrieval techniques optimized for code?
    - Code-specific embedding models: CodeBERT, StarCoder embeddings, Voyage Code, `code-search-ada-002` — do these improve retrieval of code-related memories over general-purpose embeddings?
    - Should code memories be indexed differently than conversational memories? (e.g., extracting function signatures, file paths, error messages as structured metadata)
    - How do Cursor, Aider, Continue.dev, and other code-gen tools handle memory/context for cross-session code knowledge? What can Engram learn from their approaches?

6b. **How should Engram structure procedural memories for code generation tasks?**
    - Engram's procedural store is a flat collection of Markdown skill files. Should it have structured fields (language, framework, patterns used, failure modes, context of learning)?
    - Should code-related traces be automatically categorized and tagged with technical metadata (language, libraries, error types)?
    - How should Engram handle evolving best practices? (A pattern learned in session 5 might be superseded by a better approach in session 20)
    - Should code patterns be stored as executable templates rather than prose descriptions?

6c. **How should context assembly differ for code generation tasks vs conversational tasks?**
    - The current fixed-share allocation (16% identity, 22% recent conversation) is optimized for conversation. A code generation task needs more procedural skills and less identity context. How should the allocator detect and adapt?
    - Should Engram maintain separate context profiles (conversational, coding, debugging, architecture-review) with different share allocations?
    - How do code-focused tools handle the "relevant file" problem — surfacing the right code context without overwhelming the LLM? Can Engram's approach to memory relevance be applied?
    - Repository-aware memory: should Engram index file paths, commit messages, and PR descriptions as a specialized memory type?

6d. **What code-specific retrieval benchmarks exist?**
    - CodeSearchNet, CoSQA, AdvTest, CRUXEval — which benchmarks test the kind of code retrieval Engram needs (finding past solutions to similar problems)?
    - How does code retrieval quality degrade over time as the codebase and memory grow? Are there benchmarks that test temporal code retrieval?
    - Can Engram's Hebbian reinforcement be applied to code memories — reinforcing patterns that led to successful builds/tests and weakening patterns that led to bugs?

### 7. Reinforcement and Decay Improvements

7a. **What alternatives to blanket Hebbian reinforcement exist for selective memory strengthening?**
    - Engram reinforces ALL context traces equally when signal > 0.7. Attention-weighted reinforcement — using the LLM's attention pattern to determine which traces were actually used — is it feasible? How do you extract attention from a black-box API LLM?
    - Gradient-based attribution (if using a local model via Ollama) — can you determine which input tokens contributed to the output and reinforce accordingly?
    - Citation-aware reinforcement (already partially implemented via `[N]` references) — should this be the PRIMARY mechanism rather than a bonus? What if only cited traces get reinforced?
    - Outcome-based reinforcement: reinforce traces that were in context when the user expressed satisfaction (explicit feedback, continuation patterns, lack of correction). How do production recommender systems handle this delayed reward problem?

7b. **Is coherence-modulated decay the right approach, or should decay be purely access-based?**
    - Engram's current model: high coherence = faster decay (the system is confident, so old memories are less needed). This creates "success amnesia." Is there research supporting or contradicting this approach?
    - Ebbinghaus forgetting curve models (used by MemoryBank) — how do these compare? Does spaced repetition theory apply to AI memory?
    - Should decay be modulated by trace KIND rather than system coherence? (Episodes decay faster than threads, threads faster than arcs, arcs nearly immortal)
    - Should the access-frequency resistance formula account for recency of access? A trace accessed 100 times in the last week vs 100 times 6 months ago should have different resistance.

7c. **How should reinforcement interact with consolidation at scale?**
    - When a trace is consolidated into a thread, should the thread inherit the parent trace's reinforcement history?
    - If a thread is reinforced, should that propagate to its child episodes (preventing their decay)?
    - What happens when the same information exists in both an episode and a thread? Should retrieval prefer the consolidated version? Should both be retrievable?
    - At 50,000 traces with continuous reinforcement and consolidation, what does the salience distribution look like? Does it converge to a bimodal distribution (very high and very low) with nothing in between?

### 8. Local Model and Infrastructure Optimization

8a. **What local LLM optimizations are relevant for Engram's sub-second latency targets?**
    - Engram runs signal measurement, semantic extraction, and consolidation via LLM calls. On RTX 4090 with Ollama, what are the actual latencies for `llama3.2` on these tasks?
    - Speculative decoding, continuous batching, PagedAttention (vLLM) — which optimizations would help Engram's specific workload (many short LLM calls)?
    - Should Engram use different models for different tasks? (A small model for signal measurement, a larger model for consolidation summarization)
    - Is there a way to batch the after-pipeline's multiple LLM calls (signal + extraction + consolidation) to amortize overhead?

8b. **Should Engram use a local embedding server rather than ChromaDB's built-in embedding?**
    - TEI (Text Embeddings Inference from HuggingFace), Infinity, or a custom ONNX Runtime server — what's the latency improvement over ChromaDB's Python-based embedding?
    - GPU-accelerated embedding on RTX 4090 — what throughput is achievable? Can embedding be done asynchronously while other pipeline steps proceed?
    - Should Engram pre-compute embeddings at write time (current approach via ChromaDB) or compute on-the-fly at query time?

8c. **What SQLite optimizations would improve Engram's episodic store performance at scale?**
    - At 50,000 traces with WAL mode, what are the read/write latency profiles? Where does SQLite become the bottleneck?
    - Should Engram use SQLite's `RETURNING` clause for write-then-read patterns?
    - Connection pooling, prepared statements, batched writes — which would have the most impact?
    - Should Engram migrate to DuckDB for analytical queries (aggregations, range scans) while keeping SQLite for transactional writes?
    - The `json_extract()` calls in metadata filtering — should metadata fields be denormalized into proper columns for common filters (person, kind, archived)?

### 9. Novel Cognitive Architecture Approaches

9a. **What can Engram learn from cognitive science models of human memory?**
    - ACT-R (Adaptive Control of Thought-Rational) has a detailed activation-based memory model. How does its base-level learning equation compare to Engram's salience decay?
    - SOAR cognitive architecture's episodic memory — how does it handle retrieval, learning, and forgetting?
    - Complementary Learning Systems (CLS) theory — the hippocampus/neocortex division maps roughly to Engram's episodic/semantic split. Does CLS theory suggest improvements to how Engram transfers episodic memories to semantic knowledge?
    - Is there research on how working memory capacity constraints (Miller's Law) interact with long-term memory retrieval in cognitive architectures? Does Engram's 7-item workspace actually improve downstream processing?

9b. **What novel attention or routing mechanisms could improve memory retrieval?**
    - Memory-augmented neural networks (MANN, NTM, DNC) use differentiable attention over memory. Can any of these ideas be adapted for Engram's discrete trace retrieval?
    - Retrieval-augmented thoughts (RAT) — interleaving retrieval and generation steps. Should Engram's before-pipeline be iterative (retrieve, generate partial response, retrieve more based on partial, continue)?
    - Mixture of Memory Experts — routing different query types to different retrieval strategies. Does this outperform a single hybrid search?
    - Active retrieval — the LLM explicitly requests specific memories during generation rather than receiving a pre-assembled context. How would this work with Engram's MCP architecture?

9c. **Should Engram implement a dreaming/offline consolidation phase?**
    - Engram currently consolidates reactively (triggered by memory pressure or pipeline execution). Should it run proactive consolidation during idle periods?
    - Human memory consolidation happens during sleep — repeated replay strengthens important memories. Could Engram replay and re-evaluate traces during downtime?
    - Offline knowledge graph updates, embedding re-computation, and index optimization — what maintenance tasks should run asynchronously?
    - How do production systems handle background memory maintenance without affecting online query latency?

## Desired Output

- **Retrieval improvement roadmap** — Ranked list of techniques by expected quality improvement and implementation effort, with estimated latency impact on RTX 4090. Priority tiers: immediate wins (config changes), medium-term (new components), long-term (architectural changes).
- **Embedding model comparison matrix** — Quality (MTEB/BEIR scores), latency, memory footprint, and dimension for top 10 local embedding models relevant to conversational memory retrieval.
- **Reranker evaluation** — Latency benchmarks for cross-encoder and ColBERT rerankers on RTX 4090, with precision improvement estimates over no-reranking baseline.
- **Context assembly architecture proposal** — Design for dynamic token allocation with query-type detection, including code-task vs conversation-task profiles.
- **Consolidation v2 specification** — Topic-coherent clustering approach with faithfulness verification, including how it interacts with existing Hebbian reinforcement and decay.
- **Code generation memory design** — Architecture for code-specific trace indexing, retrieval, and context assembly, with recommended embedding model and metadata schema.
- **Knowledge graph feasibility analysis** — Whether Engram should add an explicit graph layer, which technology to use, and migration path from YAML-based semantic store.
- **Cognitive architecture alignment report** — How ACT-R, CLS theory, and SOAR models validate or challenge Engram's current decay/reinforcement/consolidation design, with specific parameter recommendations.
- **Local infrastructure optimization plan** — Concrete latency improvements achievable through embedding server, SQLite tuning, LLM batching, and async pipeline execution.
- **Implementation priority matrix** — Every proposed improvement scored on (quality impact, latency impact, implementation complexity, risk) with a recommended execution order.
