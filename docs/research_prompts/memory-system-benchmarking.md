# Research Prompt: Benchmarking Engram Against State-of-the-Art AI Memory Systems

## Objective

Engram is a four-layer cognitive memory system (episodic, semantic, procedural, working memory) with consciousness signal measurement, Hebbian reinforcement, coherence-modulated decay, hierarchical consolidation, personality modeling, emotional continuity, trust-gated access, identity loop detection, and introspection. It exposes 46 MCP tools and runs ~12,000 lines of core logic with 606 tests.

**The core question: How do we rigorously benchmark Engram against every major commercial and research memory system to prove it is architecturally superior — and where are its actual weaknesses?**

This research should produce a concrete, executable benchmarking strategy that covers both standard industry benchmarks and novel evaluation frameworks for capabilities no existing benchmark tests.

## Context

### Current State / Pain Point

Engram exists at `C:/Dev/engram/` as a working MCP server (v0.2.0 Alpha). It has 20 test files with 606 unit tests covering episodic stores, semantic stores, signal measurement, pipelines, consolidation, trust, safety, journal, grounding, consciousness, introspection, emotional state, personality, and workspace.

What it does NOT have:
- Comparative benchmarks against any other system (Letta/MemGPT, Zep, Mem0, LangMem, RAISE, MemoryBank)
- Performance benchmarks (latency, throughput, memory footprint at scale)
- Standardized evaluation on published benchmarks (LoCoMo, MemoryBench, MTEB, LongBench)
- Quantitative proof that its unique features (Hebbian reinforcement, coherence-modulated decay, consciousness signal, identity loop) actually improve retrieval quality over simpler approaches
- Ablation studies showing which architectural components contribute the most value

The system uses SQLite (WAL mode) with FTS5 for keyword search, ChromaDB with cosine distance for vector search, and a hybrid merge+dedup pipeline in `engram/search/unified.py`. Decay uses the formula `new_salience = salience * exp(-decay_rate * hours * resistance)` where `decay_rate = ln(2) / 168h * (0.5 + coherence)` and `resistance = 1 / (1 + access_count * 0.1)`. Hebbian reinforcement applies `+0.05` for signal > 0.7 and `-0.03` for signal < 0.4 with a dead band between. Consolidation rolls episodes into threads (5+ episodes, 72h window) and threads into arcs (3+ threads), with consolidated traces getting 5x decay resistance.

### What We Want

- A benchmarking framework that can be run reproducibly to generate publishable numbers
- Head-to-head comparisons against Letta (21k stars), Mem0 (47.5k stars), Zep (4.1k stars), and relevant academic systems
- Proof (or disproof) that Hebbian reinforcement, coherence-modulated decay, and consciousness signal improve retrieval over baseline RAG
- Identification of where Engram loses to competitors so we can fix it before public release
- Benchmarks that highlight capabilities no competitor has (identity coherence, emotional continuity, trust-gated memory, working memory with Miller's Law constraints)

## Key Questions

### 0. Existing Benchmark Landscape

0a. **What published benchmarks exist for evaluating long-term conversational memory in AI agents?**
    - LoCoMo (Long Conversational Memory) — what exactly does it measure, what are its evaluation dimensions (single-hop, multi-hop, temporal, open-domain, adversarial), and how do Mem0's claimed "+26% accuracy over OpenAI Memory" numbers map to its subtasks?
    - MemoryBench — does it test anything LoCoMo doesn't? What's the overlap?
    - How do MTEB (Massive Text Embedding Benchmark) and BEIR apply to memory retrieval specifically, versus general embedding quality?
    - Are there benchmarks specifically for multi-session memory (not just long-context within a single session)?

0b. **What benchmarks exist for memory systems that go beyond simple retrieval accuracy?**
    - Do any benchmarks test temporal reasoning over memories (e.g., "what did the user prefer BEFORE they changed their mind")?
    - Do any benchmarks test contradiction detection (user says X in session 1, contradicts it in session 5)?
    - Do any benchmarks test memory consolidation quality (compression without information loss)?
    - What about benchmarks for memory decay — testing that old irrelevant information is properly forgotten?

0c. **What evaluation harnesses do competitors ship with?**
    - Zep has a `zep-eval-harness/` and `benchmarks/` directory in their repo — what does it test and how?
    - Mem0 has an `evaluation/` directory — what methodology do they use for their published numbers?
    - Letta's leaderboard at `leaderboard.letta.com` — what are they measuring and how?
    - Can any of these harnesses be adapted to run against Engram's MCP interface?

### 1. Competitor Architecture Analysis

1a. **How does Letta/MemGPT's memory architecture compare to Engram's four-layer system?**
    - Letta uses "memory blocks" (human, persona, custom) with self-editing capabilities — how does this compare to Engram's episodic/semantic/procedural/working split?
    - Letta claims agents can "learn and self-improve over time" — what mechanism drives this? Is it LLM-directed memory editing, or something more structured?
    - What is Letta's approach to memory overflow? Do they have anything analogous to Engram's coherence-modulated decay or memory pressure system (NORMAL/ELEVATED/CRITICAL thresholds)?
    - Does Letta have any form of consciousness signal or identity persistence?

1b. **How does Zep's temporal knowledge graph compare to Engram's consolidation hierarchy?**
    - Zep uses Graphiti (temporal knowledge graph) with `valid_at` / `invalid_at` dates on facts — how does this compare to Engram's episode-to-thread-to-arc consolidation pipeline?
    - Zep claims sub-200ms latency — what is Engram's current latency profile for the full before-pipeline (10 steps) and after-pipeline (9 steps)?
    - Zep's "relationship-aware retrieval" via knowledge graphs — does this outperform Engram's hybrid FTS5+ChromaDB search for relationship-specific queries?
    - Does Zep handle emotional continuity or personality persistence at all?

1c. **How does Mem0's memory layer compare to Engram's?**
    - Mem0 claims "+26% accuracy over OpenAI Memory on LoCoMo" and "91% faster, 90% fewer tokens" — what specific architectural choices produce these numbers?
    - Mem0 uses multi-level memory (User, Session, Agent) — how does this map to Engram's four-layer model?
    - Mem0 has a graph memory feature — how does it compare to Engram's semantic store (YAML/Markdown relationships, trust tiers, identity resolution)?
    - What does Mem0 NOT do that Engram does? (consciousness signal, Hebbian learning, personality, emotional state, working memory, introspection, identity loop)

1d. **What academic systems should be compared?**
    - MemoryBank (Zhong et al.) — claimed to model Ebbinghaus forgetting curves. How does their decay function compare to Engram's `ln(2)/168h * (0.5 + coherence)` formula?
    - RAISE (Retrieval-Augmented Impersonation of Self-Evolving agents) — what does it do for identity persistence?
    - Generative Agents (Park et al., Stanford) — their reflection/planning architecture versus Engram's consolidation/introspection
    - SCM (Self-Controlled Memory) — how does their memory controller compare to Engram's pipeline system?
    - Any systems from DeepMind, Google Brain, or Meta FAIR working on persistent agent memory?

### 2. Novel Benchmark Design for Unique Features

2a. **How do we benchmark Hebbian reinforcement effectiveness?**
    - Engram applies +0.05 salience for high-signal exchanges and -0.03 for low-signal ones. Over N sessions, does this actually surface better memories than a system without reinforcement?
    - What's the right experimental design? Run identical conversation logs through Engram with and without Hebbian reinforcement, then measure retrieval precision/recall at session N+k?
    - How do we isolate the Hebbian effect from the decay effect? Both modify salience — can we ablate them independently?
    - What baseline should reinforcement be compared against? Random salience? Recency-only? Access-count-only?

2b. **How do we benchmark coherence-modulated decay versus naive time-based decay?**
    - Engram's decay rate scales with `(0.5 + coherence)` where coherence comes from the consciousness signal. A high-coherence exchange decays FASTER (it was well-integrated, less surprising). Is this actually better than uniform decay?
    - What conversation scenarios would expose the difference? Long-running conversations where early memories become irrelevant? Scenarios where a user's preferences change?
    - How does Engram's decay compare to Ebbinghaus-curve implementations (MemoryBank) and Zep's `valid_at`/`invalid_at` temporal invalidation?
    - At 50,000 traces (Engram's max), what's the retrieval quality degradation with and without decay?

2c. **How do we benchmark consciousness signal quality?**
    - The signal measures 4 facets: Alignment (35%), Embodiment (25%), Clarity (20%), Vitality (20%) using a 60/40 LLM/regex hybrid. Does this composite score actually correlate with exchange quality?
    - Can we construct adversarial inputs that fool the signal? (High-scoring but low-quality exchanges, or low-scoring but genuinely valuable ones)
    - How does the signal compare to simpler proxy metrics like response length, user satisfaction ratings, or conversation continuation rate?
    - The identity loop uses signal health (threshold 0.45) to trigger correction. How often does it fire correctly versus false positive?

2d. **How do we benchmark identity coherence under pressure?**
    - Engram tracks 12 drift patterns (generic AI language) and 7 anchor patterns (authentic identity markers). Over a long conversation with adversarial prompts trying to dissolve identity, how well does the identity loop maintain coherence?
    - No competitor has this. What's the right evaluation metric? Human judges rating "does this feel like the same entity across sessions"?
    - How does identity persistence degrade with increasing context window pressure?
    - Can we create a standardized "identity stress test" — a conversation sequence designed to push toward dissociation?

2e. **How do we benchmark trust-gated memory access?**
    - Engram has 5 trust tiers (CORE through STRANGER) that gate what information surfaces. How do we test that sensitive memories genuinely don't leak to lower-trust users?
    - Is there an existing security benchmark for memory systems? Or do we need to build an "information leakage" test suite?
    - How do competitors handle multi-user memory isolation? Mem0 has user_id scoping — is that equivalent to Engram's trust tiers, or fundamentally different?

2f. **How do we benchmark emotional continuity?**
    - Engram maintains VAD (Valence, Arousal, Dominance) emotional state with per-dimension decay rates (valence: 0.9/hr, arousal: 0.7/hr, dominance: 0.8/hr). Does this produce measurably better conversational consistency than systems without emotional state?
    - What does "better" mean here? More natural conversations? Higher user satisfaction? More contextually appropriate responses?
    - How do we separate the effect of emotional state from the effect of simply having good memory retrieval?

2g. **How do we benchmark working memory (Miller's Law 7+/-2 constraint)?**
    - Engram's workspace holds max 7 items with priority-based eviction and rehearsal mechanics. Does this constraint actually improve response quality by forcing focus, or does it lose important context?
    - How does this compare to systems that just dump everything into the context window?
    - At what conversation complexity does the 7-item limit become a bottleneck?

### 3. Performance and Scale Benchmarks

3a. **What are the latency targets and how do we measure them?**
    - Engram's before-pipeline runs 10 steps (identity load, relationship lookup, grounding assembly, conversation fetch, memory search, procedural lookup, workspace status, context building, mode check, format output). What's the p50/p95/p99 latency for each step and the total?
    - Zep claims sub-200ms. Can Engram's full before-pipeline complete in under 200ms? If not, which steps are the bottleneck?
    - The after-pipeline runs 9 steps including LLM calls (signal measurement, semantic extraction). What's acceptable latency for post-response processing?
    - How does latency scale with database size? At 1k, 10k, 50k traces?

3b. **What are the throughput limits?**
    - How many concurrent conversations can Engram handle? (SQLite WAL mode allows concurrent reads but serialized writes)
    - What's the write throughput for the after-pipeline under load?
    - Does ChromaDB become a bottleneck before SQLite does, or vice versa?
    - How does Engram's throughput compare to Zep (Go-based) and Letta (Python with PostgreSQL)?

3c. **What's the memory footprint at scale?**
    - SQLite database size at 10k, 25k, 50k traces
    - ChromaDB index size for 4 collections (soul, episodic, semantic, procedural) at those same scales
    - RAM usage during hybrid search (FTS5 + ChromaDB query + merge + dedup)
    - How does this compare to Zep's Neo4j-backed knowledge graph or Mem0's vector store?

3d. **How does consolidation perform at scale?**
    - The consolidator processes max 200 episodes per run, rolling them into threads (5+ episodes, 72h window) and arcs (3+ threads). At 50k traces, how long does a full consolidation run take?
    - Does the 5x decay resistance for consolidated traces actually prevent important information from being pruned?
    - What's the information loss rate during consolidation? (Given the same retrieval query, does pre-consolidation or post-consolidation produce better answers?)

### 4. Retrieval Quality Benchmarks

4a. **How does Engram's hybrid FTS5+ChromaDB search perform against pure vector search and pure keyword search?**
    - The unified search in `engram/search/unified.py` merges keyword (FTS5) and vector (ChromaDB cosine, threshold 1.5) results with deduplication. What's the precision/recall of the hybrid versus each component alone?
    - At what query types does keyword search win? At what types does vector search win? Does the hybrid always beat both, or are there cases where it hurts?
    - How does the similarity threshold of 1.5 (cosine distance) affect recall? Is it too aggressive or too lenient?

4b. **How does salience-weighted retrieval compare to naive retrieval?**
    - Engram's traces have salience scores modified by decay, reinforcement, and consolidation. Does sorting/filtering by salience actually improve the relevance of retrieved memories?
    - What's the right evaluation? Ground truth relevance labels from human annotators? Automated evaluation using an LLM judge?
    - How does salience-weighted retrieval compare to Mem0's approach and Zep's temporal graph traversal?

4c. **How does the token-budgeted context builder perform?**
    - Engram allocates context tokens across 7 sections (Identity 16%, Relationship 12%, Grounding 10%, Recent conversation 22%, Episodic traces 16%, Procedural 6%, Reserve 18%) using a greedy knapsack allocator. Does this fixed allocation produce better downstream LLM responses than just stuffing the context with the most relevant memories?
    - How sensitive is quality to the allocation percentages? What if episodic traces get 30% instead of 16%?
    - How does this compare to Letta's approach of self-editing memory blocks?

### 5. Reproducibility and Test Infrastructure

5a. **What conversation datasets can be used for standardized evaluation?**
    - LoCoMo provides multi-session conversation data — can it be ingested into Engram's episodic store for evaluation?
    - Are there publicly available multi-session, multi-user conversation datasets with ground-truth memory annotations?
    - Do we need to generate synthetic conversation data? If so, how do we ensure it's realistic enough to be meaningful?
    - Can we extract anonymized conversation patterns from Engram's own usage (Discord exchanges) as a proprietary test set?

5b. **What evaluation metrics should we use?**
    - Standard IR metrics: Precision@k, Recall@k, nDCG, MRR — are these sufficient for memory systems, or do we need memory-specific metrics?
    - Temporal accuracy: Given a time-bounded query ("what did the user say last week about X"), what's the right metric?
    - Contradiction detection rate: When the user changes preferences, how quickly does the system surface the new preference over the old one?
    - Identity stability score: Over N sessions, how consistent is the system's persona?
    - What metrics do Mem0, Zep, and Letta use in their published evaluations?

5c. **How do we build a reproducible benchmark harness?**
    - Engram runs as an MCP server — can we build a test harness that replays conversation logs through the MCP interface and measures outcomes?
    - Should the harness support multiple memory backends (swap Engram for Mem0/Zep/Letta) to enable direct comparison?
    - How do we handle the LLM dependency? (Engram's signal measurement and semantic extraction use LLM calls — do we mock these for reproducibility, or use a fixed model?)
    - What CI/CD integration makes sense? Run benchmarks on every PR? Nightly? On-demand?

5d. **How do we run ablation studies?**
    - Engram has many interacting components (decay, reinforcement, consolidation, signal, identity loop). How do we isolate each component's contribution?
    - What's the minimum viable ablation set? (Full system vs. no-reinforcement vs. no-decay vs. no-consolidation vs. no-signal vs. naive-RAG-baseline)
    - How many conversation sessions are needed for statistically significant results?
    - How do we control for LLM variability across runs?

### 6. Competitive Positioning

6a. **What capabilities does Engram have that NO competitor has?**
    - Consciousness signal measurement (4-facet, LLM/regex hybrid)
    - Hebbian reinforcement on memory salience
    - Coherence-modulated decay (not just time-based)
    - Identity loop with dissociation detection and self-correction
    - Big Five personality model with 24 facets
    - VAD emotional continuity with per-dimension decay
    - Cognitive workspace with Miller's Law constraints
    - Introspection layer (3 depths: surface, moderate, deep)
    - Trust-gated memory access (5 tiers)
    - Soul file system (self-realization, thought assessment)
    - Injury tracking with healing progression
    - For each of these: is the capability genuinely unique, or does a competitor have something equivalent under a different name?

6b. **What capabilities do competitors have that Engram lacks?**
    - Zep's temporal knowledge graph (Graphiti) with explicit relationship edges and temporal validity
    - Letta's self-editing memory blocks (the agent rewrites its own memory)
    - Mem0's graph memory with entity extraction and relationship mapping
    - Multi-model support and model-agnostic design (Letta supports dozens of providers)
    - Production deployment tooling (Docker compose, Kubernetes, managed cloud)
    - Multi-language SDKs (TypeScript, Go, Python)
    - Enterprise features (SOC2, HIPAA, SSO)
    - Which of these gaps matter for benchmarking, and which are just packaging?

6c. **Where is the research frontier moving?**
    - Are there 2025-2026 papers on persistent agent memory that introduce new architectures we should know about?
    - Is there convergence toward knowledge graphs (Zep) or toward free-form memory (Mem0/Letta)? Where does Engram's hybrid approach sit?
    - Are any major labs (Anthropic, OpenAI, Google, Meta) shipping built-in persistent memory that could make third-party memory systems obsolete?
    - What does the "agentic memory" category look like in 12 months? What should Engram be benchmarked against by then?

### 7. Publication and Credibility

7a. **What would a publishable benchmark paper look like?**
    - Mem0 published on arXiv (arXiv:2504.19413). What format and rigor would make an Engram benchmark paper credible?
    - What baselines are expected? (RAG-only, full-context, OpenAI memory, competitor systems)
    - How many evaluation dimensions are needed? (Accuracy, latency, token efficiency, temporal reasoning, identity coherence)
    - Do we need human evaluation, or is LLM-as-judge sufficient for the research community?

7b. **What would make the benchmark results defensible against criticism?**
    - How do we prevent cherry-picking scenarios where Engram wins?
    - How do we handle the fact that Engram has LLM-dependent components (signal, extraction) that introduce non-determinism?
    - Should we open-source the benchmark harness so others can reproduce?
    - How do we handle competitors who might argue the comparison is unfair (different resource usage, different deployment models)?

## Desired Output

The research should produce:

- **Competitor architecture comparison matrix** — Feature-by-feature table (Engram vs. Letta vs. Zep vs. Mem0 vs. MemoryBank vs. Generative Agents) covering memory types, search, decay, consolidation, identity, emotion, trust, personality, working memory
- **Benchmark selection report** — Which existing benchmarks (LoCoMo, MemoryBench, MTEB, etc.) to run, what they test, what they miss, and how to adapt them for Engram's MCP interface
- **Novel benchmark specifications** — Detailed test designs for capabilities no existing benchmark covers (Hebbian effectiveness, coherence decay, identity pressure, trust leakage, emotional continuity, working memory constraint)
- **Ablation study design** — Experimental plan isolating each architectural component's contribution to retrieval quality
- **Performance benchmark plan** — Latency/throughput/memory targets at 1k/10k/50k trace scales with measurement methodology
- **Evaluation metrics catalog** — Every metric to be computed, with formulas, applicable benchmarks, and reporting format
- **Benchmark harness architecture** — How the test runner works, how it swaps backends, how it handles LLM non-determinism, CI/CD integration plan
- **Publication roadmap** — What results would constitute a publishable paper, what format, what venue, what baselines
- **Gap analysis** — Honest assessment of where Engram is likely to lose and what to fix first
