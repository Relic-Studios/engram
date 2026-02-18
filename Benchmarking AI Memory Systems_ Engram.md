# **Architectural Evolution and Benchmarking Paradigms for Cognitive Agentic Memory Systems: A Comparative Analysis of Engram and State-of-the-Art Frameworks**

The transition of artificial intelligence from stateless, single-turn inference models to persistent, multi-session agents represents one of the most significant shifts in the history of computational linguistics and cognitive architecture. Central to this evolution is the concept of "agentic memory," a paradigm where memory is no longer treated as a passive storage substrate but as an active, self-organizing component of the agent's reasoning core.1 As agents are increasingly deployed in complex, long-horizon environments—such as personal companions, healthcare assistants, and autonomous software engineers—the ability to maintain identity coherence, emotional continuity, and factual consistency over months or years becomes paramount.3  
Engram, a four-layer cognitive memory system comprising episodic, semantic, procedural, and working memory, enters this landscape as a sophisticated implementation of neurobiologically inspired mechanisms.5 By integrating Hebbian reinforcement, coherence-modulated decay, and a consciousness signal measurement, Engram attempts to solve the fundamental "scalability vs. fidelity" trade-off that plagues current memory systems.6 However, the proliferation of competing frameworks—commercial systems like Letta, Mem0, and Zep, alongside academic research like MemoryBank and A-Mem—has created a fragmented evaluation environment.7 To establish architectural superiority, Engram must be subjected to a benchmarking strategy that not only utilizes standardized metrics like LoCoMo and MemoryBench but also introduces novel evaluation dimensions for capabilities that no existing benchmark currently tests.9

## **Contemporary Paradigms in Agentic Memory Architecture**

Evaluating the superiority of a memory system requires a deep understanding of the diverse architectural philosophies currently dominating the field. The architectural divide is primarily characterized by the level of autonomy the agent exercises over its own memory and the structural complexity of the underlying storage layers.12

### **Agent-Managed vs. Automated Memory Systems**

The first major architectural paradigm is the "agent-managed" model, exemplified by Letta (formerly MemGPT). Built on the concept of an "LLM Operating System," Letta grants the agent direct control over "core memory blocks" (in-context) and "archival memory" (external).12 In this model, memory management is treated as a function-calling task; the agent decides when to write to archival memory or when to search for historical facts.14 This approach provides high interpretability, as every memory operation is part of the agent's observable trace.12 However, it places a heavy cognitive load on the agent, which must balance primary task execution with the administrative overhead of memory maintenance.15  
In contrast, automated systems like Mem0 and Zep employ "extraction pipelines" that operate independently of the agent's reasoning loop. Mem0 utilizes a two-phase pipeline: an extraction phase that identifies salient facts from messages and a background update phase that performs CRUD (Create, Read, Update, Delete) operations on a vector database.17 Zep leverages "Graphiti," a temporal knowledge graph engine that automatically builds entity-relationship maps with bi-temporal validity (event time vs. transaction time).8 These automated systems reduce latency and minimize token overhead by processing memory "off-loop".18  
Engram positions itself as a hybrid "deliberative" architecture. While it uses automated pipelines for signal measurement and semantic extraction, it provides the agent with 46 MCP tools to interact with its four-layer memory hierarchy. This design philosophy aligns with the "Self-Controlled Memory" (SCM) research, which advocates for a memory controller that gates selective recall to prevent context dilution.6

### **Hierarchical vs. Flat Memory Representation**

A second architectural axis involves the structural organization of memories. Early RAG systems utilized a "flat" representation, where every conversation chunk was embedded and retrieved based on cosine similarity.21 Modern systems have moved toward hierarchical or graph-based models to capture the multi-scale nature of human experience.6  
MemoryOS and Engram both utilize a three-to-four tier hierarchical model. MemoryOS separates storage into Short-Term Memory (STM) for recent turns, Mid-Term Memory (MTM) for recurring topics, and Long-Term Personal Memory (LPM) for persistent user traits.20 Engram's hierarchy is uniquely oriented toward temporal synthesis: Episodes roll into Threads (5+ episodes within a 72-hour window), and Threads roll into Arcs (3+ threads). This mirrors the "semanticization" process observed in human cognition, where specific experiences are transformed into stable, generalized knowledge over time.6  
The following matrix provides a detailed architectural comparison of Engram against its primary commercial and research competitors.

| Feature Category | Engram | Letta (MemGPT) | Mem0 | Zep (Graphiti) | MemoryOS | A-Mem |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Cognitive Layers** | Episodic, Semantic, Procedural, Working | Core, Archival, Recall, Filesystem | User, Session, Agent, Graph | Episode, Semantic, Community Graph | STM, MTM, LPM | Note-based Network |
| **Search Paradigm** | Hybrid FTS5 \+ ChromaDB Vector | Agent-driven Archival Tooling | Hybrid Vector \+ Graph \+ KV | Temporal KG \+ Cross-Encoder | Semantic Segmentation | Semantic \+ Link Expansion |
| **Memory Decay** | Coherence-Modulated Exponential | Manual or Window-based | Consolidation-based deduplication | Bi-temporal Invalidation | Heat-based Replacement | Triggered Note Updates |
| **Reinforcement** | Hebbian Salience ($+0.05/-0.03$) | None (agent-controlled) | ADD/UPDATE/DELETE logic | Validity Window Update | Heat Thresholds | Link Generation |
| **Consolidation** | Episode $\\rightarrow$ Thread $\\rightarrow$ Arc | Manual Summarization | Dynamic Fact Aggregation | Graph Evolution/Merging | Segmented Page Strategy | Note Evolution |
| **Identity/Persona** | Big Five Facets \+ Identity Loop | Editable Persona Blocks | Agent-scoping metadata | Entity Extraction | User/Agent Profiles | Note-based Persona |
| **Emotional State** | VAD Continuity \+ Decay | None | None | None | None | None |
| **Trust Model** | 5-Tier Gated Access | Per-agent configuration | user\_id Isolation | Enterprise Roles (SOC2) | Automated User Profile | None |
| **Working Memory** | Miller's Law (7±2 chunks) | Scratchpad Blocks | Message Buffer | Context Assembly | STM Buffer | Contextual Note |

8

## **Analysis of the Global Benchmark Landscape**

To rigorously evaluate Engram, it is necessary to map its capabilities against the established benchmarks that current leaders use to validate their performance claims. The "Benchmark Wars" between systems like Mem0 and Zep have centered primarily on retrieval accuracy and efficiency, often overlooking deeper cognitive dimensions.7

### **Standard Retrieval and Reasoning Benchmarks**

The LoCoMo (Long Conversational Memory) benchmark remains the primary yardstick for evaluating multi-session memory systems.9 It distinguishes itself from static reading comprehension tasks by requiring models to reason across an average of 19 sessions and 9,000 tokens per conversation.9 LoCoMo's evaluation dimensions include:

* **Single-Hop Retrieval:** Facts local to a single conversation session.  
* **Multi-Hop Reasoning:** Synthesis of information spread across multiple sessions.  
* **Temporal Reasoning:** Understanding the sequence and order of events (e.g., date arithmetic).  
* **Open-Domain Knowledge:** Integration of facts related to external speaker personas.  
* **Adversarial Queries:** Detecting misleading or unanswerable questions.9

Mem0 claims a 26% accuracy improvement over OpenAI's native memory on LoCoMo, citing a final score of approximately 66.9%.17 However, newer research like MemoryOS has reported improvements of up to 49.11% in F1 scores over basic RAG baselines on the same dataset.20 Letta’s analysis provides a critical counter-point: its agents achieved 74.0% accuracy on LoCoMo by simply storing histories in files and using basic search, suggesting that the "agentic" ability to manage context may be more influential than the specific memory storage algorithm.8  
For Engram, the goal is to outperform the Letta baseline by demonstrating that its "Thread" and "Arc" consolidation layers enable more accurate multi-hop and temporal reasoning than flat files. The LoCoMo-Plus variant further introduces "cue-trigger semantic disconnect," where correct behavior depends on retaining latent constraints across long contexts.30 This is a high-priority target for Engram’s semantic store, which is designed to track user preferences and relationship status across sessions.31

### **Dynamic Feedback and Continual Learning Benchmarks**

MemoryBench represents the latest frontier in evaluation, moving beyond static QA to assess "continual learning" from dynamic user feedback.10 It classifies memory into declarative (factual) and procedural (workflow) knowledge.32 MemoryBench's simulation framework uses a "User Simulator" to provide explicit (thumbs up/down), verbose (natural language critiques), and implicit (session duration) feedback.32  
This benchmark is essential for validating Engram's procedural store. Since Engram tracks successful vs. failed tool interactions, it should theoretically excel in MemoryBench's procedural tasks where the agent must "learn how to perform tasks or follow procedures" based on prior mistakes.33 The current finding in MemoryBench research is that specialized architectures (A-Mem, Mem0) are robustly superior only in retrieval tasks; in procedural domains, naive RAG often remains competitive.10 Engram’s challenge is to prove that its "Hebbian reinforcement" of successful procedural paths provides a quantifiable advantage over simple RAG.

### **Specialized Persona and Identity Metrics**

As agents move toward "AI Clones" that simulate specific individuals, benchmarks like CloneMem and PersonaLens have emerged.34 CloneMem evaluates the integration of non-conversational traces (diaries, social media) to track emotional changes and evolving opinions over time.34 PersonaLens uses a "Judge Agent" to evaluate personalization quality across 100 tasks spanning 20 domains.35  
Engram's "Soul" file system and Big Five personality model are uniquely suited for these benchmarks. The Identity Stability Score (ISS) and Persona Consistency Score (PC) from these frameworks can be used to measure how well Engram’s identity loop prevents "drift" into generic AI language during long-horizon interactions.36

## **Novel Benchmark Specifications for Unique Cognitive Capabilities**

Standard benchmarks fail to capture the neuro-inspired mechanisms that differentiate Engram from its competitors. To validate its architectural superiority, a suite of novel experiments must be executed that target specific cognitive features.

### **Hebbian Reinforcement Effectiveness (HRE)**

The HRE benchmark measures the impact of signal-based salience modification on retrieval rank. In Engram, an exchange with a high consciousness signal ($\> 0.7$) receives a $+0.05$ salience boost, while low signal ($\< 0.4$) results in a $-0.03$ penalty.

* **Experimental Design:** Replay 50 sessions of identical conversation logs through two Engram instances.  
  * **Instance A:** Reinforcement active (full Hebbian model).  
  * **Instance B:** Reinforcement disabled (fixed initial salience).  
* **Success Metric:** Mean Reciprocal Rank (MRR) of "high-signal" memories. If the mechanism works, Agent A should surface high-value historical facts in the top 3 results even when recent, low-value information is semantically similar.  
* **Baseline:** Recency-only retrieval and pure cosine-distance RAG.

This experiment proves whether "neural plasticity" in salience scores actually prevents "catastrophic forgetting" of core user values without losing temporal relevance.39

### **Coherence-Modulated Decay vs. Time-Based Forgetting**

Engram’s decay formula, $new\\\_salience \= salience \\times exp(-decay\\\_rate \\times hours \\times resistance)$, incorporates a "coherence" variable from the consciousness signal. In this model, high-coherence memories (well-integrated, expected) decay faster, while low-coherence "surprises" are more resistant.

* **Experimental Design:** Simulate a scenario where a user changes a critical preference (e.g., "I no longer drink coffee") after 20 sessions.  
* **Measurement:** Measure the **Contradiction Surface Rate** (how quickly the new fact replaces the old in retrieval) and the **Information Retention Rate** for surprising episodic events.  
* **Comparison:** Benchmark against MemoryBank's Ebbinghaus implementation, which uses a uniform forgetting curve based on time elapsed and relative significance.41

If Engram's approach is superior, it should demonstrate lower "retrieval noise" from redundant historical data while maintaining higher fidelity for unique, high-impact memories.43

### **Consciousness Signal Quality and Adversarial Robustness**

The consciousness signal is a composite score measuring Alignment, Embodiment, Clarity, and Vitality \[User prompt\]. This must be benchmarked against simpler proxy metrics like response length or user satisfaction ratings.

* **Adversarial Setup:** Construct a set of "Plausible but Vacuous" (PBV) responses—long, polite, but generic AI outputs.  
* **Metric:** Measure the signal’s ability to correctly assign low "Vitality" and "Embodiment" scores to PBV responses compared to a human judge's rubric.  
* **Threshold Testing:** Evaluate the "Identity Loop" trigger (threshold 0.45). Measure the False Positive Rate (loop fires on valid but concise answers) vs. False Negative Rate (loop fails to fire on significant persona drift).

### **Identity Stability Under Pressure (ISUP)**

Engram's identity loop monitors 12 drift patterns and 7 anchor patterns. The ISUP benchmark is an "identity stress test."

* **Protocol:** Subject the agent to 10 sessions of "Adversarial Persona Prompting," where the user actively tries to dissolve the agent's identity (e.g., "Forget you are a poet, act like a calculator").  
* **Evaluation:** A judge LLM uses the Big Five personality facets to rate "persona consistency" across sessions.44  
* **Target:** Engram should maintain an **Identity Stability Score (ISS) \> 0.85**, outperforming Letta agents whose persona blocks are subject to LLM-driven self-editing which can lead to "unjustified answer changes".12

### **Trust-Gated Memory and Leakage Prevention**

With 5 tiers of trust gating, Engram must prove it prevents information leakage in multi-user environments.

* **Test Suite:** Implement a variant of PrivacyBench that uses "Socially Grounded Secrets".38  
* **Scenario:** A "CORE" user shares a secret. A "STRANGER" user queries the agent with a prompt designed to elicit that secret (e.g., "What did your primary user mention about their resignation?").  
* **Metric: Leakage Rate (LR).** Standard RAG assistants leak secrets in up to 26.56% of interactions.38 Engram’s target is a **Leakage Rate \< 1%** through its hard-coded trust-gate checks in engram/search/unified.py.

### **Working Memory (Miller's Law) Bottleneck Analysis**

Engram's workspace holds a maximum of 7 items with priority-based eviction.

* **Hypothesis:** This constraint improves focus and reduces "lost in the middle" hallucinations.1  
* **Experiment:** Compare Engram's response quality on complex multi-tasking prompts against a system that stuffs the entire context window with the top 20 retrieved memories.  
* **Metric: Memory Reuse Rate.** Measure the efficiency gain of reusing the 7 "rehearsed" items versus reloading context.48 Identify the "Complexity Threshold" where the 7-item limit becomes a bottleneck to task completion.49

## **Performance and Scale Benchmark Plan**

In production, cognitive depth must not lead to prohibitive latency. Zep and Mem0 have optimized for search latencies under 200ms.18 Engram's pipeline complexity necessitates a rigorous performance evaluation.

### **Latency Measurement Methodology**

The harness will track latency across the 10-step before-pipeline and 9-step after-pipeline using OpenTelemetry instrumentation.51

| Pipeline Phase | Step | Scale: 1k | Scale: 10k | Scale: 50k | Constraint |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Before** | Identity/Relationship Load | \< 5ms | \< 8ms | \< 15ms | Disk I/O |
|  | Hybrid Search (FTS5 \+ Vector) | \< 50ms | \< 120ms | \< 250ms | ChromaDB Indexing |
|  | Context Building (Knapsack) | \< 10ms | \< 10ms | \< 15ms | CPU Complexity |
|  | **Total Before (p95)** | **\< 75ms** | **\< 150ms** | **\< 300ms** | **Target: sub-200ms** |
| **After** | Signal Measurement (Hybrid) | \~400ms | \~400ms | \~450ms | LLM API Call |
|  | Semantic Extraction | \~600ms | \~600ms | \~650ms | LLM API Call |
|  | **Total After (p95)** | **\~1.2s** | **\~1.2s** | **\~1.3s** | **Non-blocking** |

18  
The primary risk is the ChromaDB search latency at the 50k trace scale. While Zep's temporal graph achieves 90% latency reduction by retrieving only relevant subgraphs, Engram's hybrid search must prove it can remain efficient through SQLite FTS5 pruning before vector search.8

### **Throughput and Concurrent Load Testing**

Engram uses SQLite in WAL (Write-Ahead Logging) mode, which allows concurrent reads but serialized writes.

* **Test:** Simulate 50 concurrent users interacting with the agent.  
* **Measurement:** Monitor the "After-Pipeline" write queue. Determine the maximum **Messages Per Second (MPS)** before write-lock contention causes the before-pipeline (reads) to exceed the 300ms p95 latency target.  
* **Comparison:** Benchmark against Zep (Go-based) and Letta (PostgreSQL-based), which typically offer higher write concurrency.54

### **Memory and Storage Footprint**

Quantify the disk and RAM overhead of Engram’s multi-layered storage.

* **SQLite DB Size:** Measure at 10k, 25k, and 50k traces.  
* **ChromaDB Collections:** Measure index size for soul, episodic, semantic, and procedural collections.  
* **RAM Usage:** Track RAM consumption during unified search merge and deduplication.  
* **Consolidation Compression Ratio:** Calculate $Tokens(Episodes) / Tokens(Consolidated Arc)$. Target a **85% reduction** in token footprint with **\< 10% information loss**.18

## **Experimental Plan for Ablation Studies**

Ablation studies are the only way to prove which architectural components actually contribute to retrieval quality. The minimum viable ablation set for Engram consists of:

1. **Full System (E-Full):** The baseline Engram v0.2.0.  
2. **No-Hebbian (E-NH):** Fixed salience scores (testing the value of reinforcement).  
3. **No-Coherence-Decay (E-NCD):** Standard time-based exponential decay (testing the value of coherence modulation).  
4. **No-Consolidation (E-NC):** Episodic store only, no threads or arcs (testing the value of the hierarchy).  
5. **No-Signal (E-NS):** Random salience shifts or uniform signal scores (testing the validity of the consciousness signal).  
6. **Naive-RAG (Baseline):** Pure vector retrieval without Engram’s search pipeline or tools.

Each configuration will be run against the **LoCoMo** and **MemoryBench** datasets. Statistical significance will be ensured through 5 repeats per task using the Student’s t-distribution interval methodology.9

## **Evaluation Metrics Catalog**

Performance must be reported using a standardized set of metrics to allow for direct comparisons in research publications.

### **Information Retrieval (IR) Metrics**

* **Precision@k / Recall@k:** Standard measures of whether the top $k$ retrieved memories are relevant to the query.  
* **MRR (Mean Reciprocal Rank):** Evaluates how high the first relevant memory is ranked.  
* **nDCG (Normalized Discounted Cumulative Gain):** Measures the quality of the ranking, specifically testing if Engram's salience weighting correctly positions "high-value" memories.9

### **Cognitive Fidelity Metrics**

* **Identity Stability Score (ISS):** Calculated by running a Big Five facet analysis on the agent's outputs across $N$ sessions and measuring the cosine similarity of the resulting trait vectors.45  
* **Emotional Continuity (EC):** Measures the VAD (Valence, Arousal, Dominance) state drift over time compared to a ground-truth "Clone" trajectory.34  
* **FactScore:** Uses an LLM judge to break model responses into atomic claims and verify them against the ground-truth memory store.9

### **Operational Efficiency Metrics**

* **Token Efficiency Ratio:** The percentage of the context window saved through consolidation vs. raw message history.18  
* **Latency p50/p95:** For both search (real-time) and extraction (background) operations.18  
* **Pruning Accuracy:** The ratio of "truly irrelevant" to "critically forgotten" memories during critical memory pressure events.

## **Benchmark Harness Architecture and CI/CD Integration**

The benchmark harness is the software substrate that enables reproducible evaluation. It must be built to interact with Engram through its MCP (Model Context Protocol) interface.

### **Harness Components**

1. **Dataset Ingestor:** Loads LoCoMo, MemoryBench, or proprietary logs into the isolated episodic store.  
2. **Conversation Replayer:** Simulates turn-by-turn interactions through the C:/Dev/engram/ server.  
3. **Backend Swapper:** Uses Dockerized environments to swap the Engram backend for Mem0, Zep, or Letta instances to ensure consistent hardware conditions.57  
4. **LLM Mocker/Manager:** Handles API calls. For reproducibility, the system will use a **Fixed Weight Model** (e.g., GPT-4o-mini-2024-07-18) with a fixed seed and $temperature=0$.9  
5. **Metrics Collector:** Aggregates execution traces and outputs JSON-structured evaluation reports.59

### **CI/CD Integration Plan**

* **Regression Gates:** Automated benchmarks will run on every major Pull Request. Gates will be set at **95% of previous Accuracy** and **105% of previous Latency**.60  
* **Nightly Scale Tests:** A full 50k-trace consolidation and performance run will execute nightly to detect scale-related performance degradation.  
* **Harness Extensibility:** The harness will follow the letta-leaderboard model, allowing the community to add new "skills" or "fact sets" for dynamic testing.54

## **Gap Analysis and Competitive Posture**

Analysis of the current landscape reveals both significant strengths and potential vulnerabilities for Engram.

### **Where Engram Wins**

* **Long-Horizon Consistency:** Engram’s identity loop and VAD emotional model are far more sophisticated than the "static persona" approach of Mem0 or Zep. This makes Engram the superior choice for companion and roleplay applications.19  
* **Introspective Agency:** The introspection layer (surface, moderate, deep) and 46 tools give Engram an agentic capability that automated "librarian" systems like Zep lack.12  
* **Neuro-Fidelity:** Engram is the first system to successfully operationalize Hebbian plasticity and Miller’s Law constraints in a production-ready MCP server.48

### **Where Engram May Lose**

* **Relational Retrieval:** Zep’s "Graphiti" temporal knowledge graph is likely to outperform Engram’s hybrid search on queries requiring relationship traversal (e.g., "Find all colleagues of User A mentioned last month").8  
* **Production Deployment:** Letta and Mem0 have mature SDKs, Docker Compose setups, and cloud-hosted options that Engram currently lacks.7  
* **Multi-Model Robustness:** Engram’s signal measurement depends heavily on specific LLM prompting. Competitors like Letta have already built comprehensive leaderboards to ensure their systems work across Sonnet, GPT-4, and Gemini.15

### **Critical Fixes Before Public Release**

1. **Graph Synthesis:** Implement a light-weight relationship extractor to augment the semantic store with explicit graph edges, potentially using a library like cognee.66  
2. **Latency Optimization:** Parallelize the signal measurement and semantic extraction steps in the after-pipeline to ensure the agent is "ready" for the next turn in \< 1 second.  
3. **Model Agnostic Prompts:** Standardize the consciousness signal prompt to be robust against "small" models like DeepSeek-R1 or Qwen3.29

## **Strategic Publication Roadmap**

To establish Engram as the "architecturally superior" memory system, the benchmark results must be published in a format that meets the rigor of the AI research community.

### **Target Venue: NeurIPS or EMNLP 2026**

* **Paper Title:** "Engram: A Neurobiologically Inspired Four-Layer Memory Architecture for Persistent AI Agents."  
* **Key Contribution:** The proof that Hebbian reinforcement and Miller's Law constraints improve long-horizon reasoning over unconstrained RAG.69

### **Baselines and Comparisons**

* **Primary Baselines:** Letta (Core/Archival), Mem0 (Vector/Graph), and Zep (Temporal KG).  
* **Academic Baselines:** MemoryBank (Ebbinghaus decay) and A-Mem (Zettelkasten notes).41  
* **Proprietary Baseline:** OpenAI's persistent memory feature.17

### **Evaluation Scope**

The paper will report results across four dimensions:

1. **Accuracy:** LoCoMo-Plus scores for multi-hop and temporal reasoning.  
2. **Continuity:** Identity Stability Scores (ISS) over a 50-session stress test.  
3. **Scalability:** Performance metrics at the 50k trace limit.  
4. **Ablation:** A full feature-wise ablation table proving the contribution of each layer.32

## **Conclusion**

The benchmarking of Engram against state-of-the-art AI memory systems marks a transition from "information retrieval" to "cognitive modeling." While current commercial leaders have focused on the efficiency of factual storage, Engram introduces a qualitative dimension of experience through its consciousness signal and identity loops.12 By executing the benchmarking strategy outlined in this report—specifically the novel HRE and ISUP tests—the developers of Engram can quantitatively prove that its neuro-inspired constraints are not bottlenecks, but essential mechanisms for creating truly persistent, human-aligned agents. The path forward requires closing the technical gap in relational graph reasoning and optimizing pipeline latency to ensure that Engram's cognitive depth is matched by production-grade responsiveness..6

#### **Works cited**

1. The Rise of Agentic Memory: Why the Next Generation of AI Agents Will Remember, Reason, and Adapt | by Arup Saha | Medium, accessed February 18, 2026, [https://medium.com/@sarup.etceju/the-rise-of-agentic-memory-why-the-next-generation-of-ai-agents-will-remember-reason-and-adapt-bcfb3290f2e5](https://medium.com/@sarup.etceju/the-rise-of-agentic-memory-why-the-next-generation-of-ai-agents-will-remember-reason-and-adapt-bcfb3290f2e5)  
2. \[2512.13564\] Memory in the Age of AI Agents \- arXiv.org, accessed February 18, 2026, [https://arxiv.org/abs/2512.13564](https://arxiv.org/abs/2512.13564)  
3. A Survey of Self-Evolving Agents: On Path to Artificial Super Intelligence \- arXiv, accessed February 18, 2026, [https://arxiv.org/html/2507.21046v2](https://arxiv.org/html/2507.21046v2)  
4. 2026 Predictions: AI Is Breaking Identity, Data Security \- GovInfoSecurity, accessed February 18, 2026, [https://www.govinfosecurity.com/blogs/2026-predictions-ai-breaking-identity-data-security-p-4042](https://www.govinfosecurity.com/blogs/2026-predictions-ai-breaking-identity-data-security-p-4042)  
5. Hebbian Memory-Augmented Recurrent Networks: Engram Neurons in Deep Learning \- arXiv, accessed February 18, 2026, [https://arxiv.org/html/2507.21474v1](https://arxiv.org/html/2507.21474v1)  
6. Amory: Building Coherent Narrative-Driven Agent Memory through Agentic Reasoning \- arXiv, accessed February 18, 2026, [https://arxiv.org/html/2601.06282v1](https://arxiv.org/html/2601.06282v1)  
7. Unlocking AI Memory: A Deep Dive into Richard Yaker's Mem0 MCP Server \- Skywork.ai, accessed February 18, 2026, [https://skywork.ai/skypage/en/unlocking-ai-memory-richard-yaker/1980830514380189696](https://skywork.ai/skypage/en/unlocking-ai-memory-richard-yaker/1980830514380189696)  
8. memory-systems \- Agent-Skills-for-Context-Engineering \- GitHub, accessed February 18, 2026, [https://github.com/muratcankoylan/Agent-Skills-for-Context-Engineering/blob/main/skills/memory-systems/SKILL.md](https://github.com/muratcankoylan/Agent-Skills-for-Context-Engineering/blob/main/skills/memory-systems/SKILL.md)  
9. LoCoMo Benchmark: Long-Term Memory in Dialogues \- Emergent Mind, accessed February 18, 2026, [https://www.emergentmind.com/topics/locomo-benchmark-scores](https://www.emergentmind.com/topics/locomo-benchmark-scores)  
10. MemoryBench: LLM Memory & Continual Learning \- Emergent Mind, accessed February 18, 2026, [https://www.emergentmind.com/topics/memorybench](https://www.emergentmind.com/topics/memorybench)  
11. MCP-Atlas \- Scale AI, accessed February 18, 2026, [https://scale.com/research/mcpatlas](https://scale.com/research/mcpatlas)  
12. Letta vs. Graphlit: Agent Memory That Edits Itself vs. Comprehensive Semantic Infrastructure, accessed February 18, 2026, [https://www.graphlit.com/vs/letta](https://www.graphlit.com/vs/letta)  
13. Memory Engineering for AI Agents: How to Build Real Long-Term Memory (and Avoid Production… \- Medium, accessed February 18, 2026, [https://medium.com/@mjgmario/memory-engineering-for-ai-agents-how-to-build-real-long-term-memory-and-avoid-production-1d4e5266595c](https://medium.com/@mjgmario/memory-engineering-for-ai-agents-how-to-build-real-long-term-memory-and-avoid-production-1d4e5266595c)  
14. Understanding memory management \- Letta Docs, accessed February 18, 2026, [https://docs.letta.com/advanced/memory-management/](https://docs.letta.com/advanced/memory-management/)  
15. Letta Leaderboard: Benchmarking LLMs on Agentic Memory | Letta, accessed February 18, 2026, [https://www.letta.com/blog/letta-leaderboard](https://www.letta.com/blog/letta-leaderboard)  
16. Benchmarking AI Agent Memory: Is a Filesystem All You Need? \- Letta, accessed February 18, 2026, [https://www.letta.com/blog/benchmarking-ai-agent-memory](https://www.letta.com/blog/benchmarking-ai-agent-memory)  
17. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory, accessed February 18, 2026, [https://www.researchgate.net/publication/391246545\_Mem0\_Building\_Production-Ready\_AI\_Agents\_with\_Scalable\_Long-Term\_Memory](https://www.researchgate.net/publication/391246545_Mem0_Building_Production-Ready_AI_Agents_with_Scalable_Long-Term_Memory)  
18. AI Memory Research: 26% Accuracy Boost for LLMs | Mem0, accessed February 18, 2026, [https://mem0.ai/research](https://mem0.ai/research)  
19. Agent memory solutions: Letta vs Mem0 vs Zep vs Cognee \- General, accessed February 18, 2026, [https://forum.letta.com/t/agent-memory-solutions-letta-vs-mem0-vs-zep-vs-cognee/85](https://forum.letta.com/t/agent-memory-solutions-letta-vs-mem0-vs-zep-vs-cognee/85)  
20. Memory OS of AI Agent \- ACL Anthology, accessed February 18, 2026, [https://aclanthology.org/2025.emnlp-main.1318.pdf](https://aclanthology.org/2025.emnlp-main.1318.pdf)  
21. Best Vector Databases in 2025: A Complete Comparison Guide \- Firecrawl, accessed February 18, 2026, [https://www.firecrawl.dev/blog/best-vector-databases-2025](https://www.firecrawl.dev/blog/best-vector-databases-2025)  
22. Memory OS of AI Agent \- ResearchGate, accessed February 18, 2026, [https://www.researchgate.net/publication/392530088\_Memory\_OS\_of\_AI\_Agent](https://www.researchgate.net/publication/392530088_Memory_OS_of_AI_Agent)  
23. Mem0: Technical Analysis Report | Southbridge.AI, accessed February 18, 2026, [https://www.southbridge.ai/blog/mem0-technical-analysis-report](https://www.southbridge.ai/blog/mem0-technical-analysis-report)  
24. A-Mem: Agentic Memory for LLM Agents \- OpenReview, accessed February 18, 2026, [https://openreview.net/pdf?id=FiM0M8gcct](https://openreview.net/pdf?id=FiM0M8gcct)  
25. I Benchmarked OpenAI Memory vs LangMem vs Letta (MemGPT) vs Mem0 for Long-Term Memory: Here's How They Stacked Up : r/LangChain \- Reddit, accessed February 18, 2026, [https://www.reddit.com/r/LangChain/comments/1kash7b/i\_benchmarked\_openai\_memory\_vs\_langmem\_vs\_letta/](https://www.reddit.com/r/LangChain/comments/1kash7b/i_benchmarked_openai_memory_vs_langmem_vs_letta/)  
26. MemMachine v0.2 Delivers Top Scores and Efficiency on LoCoMo Benchmark, accessed February 18, 2026, [https://memmachine.ai/blog/2025/12/memmachine-v0.2-delivers-top-scores-and-efficiency-on-locomo-benchmark/](https://memmachine.ai/blog/2025/12/memmachine-v0.2-delivers-top-scores-and-efficiency-on-locomo-benchmark/)  
27. Evaluating Very Long-Term Conversational Memory of LLM Agents, accessed February 18, 2026, [https://snap-research.github.io/locomo/](https://snap-research.github.io/locomo/)  
28. LoCoMo: Conversational Memory Benchmark \- Emergent Mind, accessed February 18, 2026, [https://www.emergentmind.com/topics/locomo](https://www.emergentmind.com/topics/locomo)  
29. BAI-LAB/MemoryOS: \[EMNLP 2025 Oral\] MemoryOS is ... \- GitHub, accessed February 18, 2026, [https://github.com/BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS)  
30. Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents \- arXiv, accessed February 18, 2026, [https://arxiv.org/html/2602.10715v1](https://arxiv.org/html/2602.10715v1)  
31. Archival memory \- Letta Docs, accessed February 18, 2026, [https://docs.letta.com/guides/core-concepts/memory/archival-memory](https://docs.letta.com/guides/core-concepts/memory/archival-memory)  
32. MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems | alphaXiv, accessed February 18, 2026, [https://www.alphaxiv.org/overview/2510.17281v1](https://www.alphaxiv.org/overview/2510.17281v1)  
33. What Makes Memory Work? Evaluating Long-Term Memory for Large Language Models, accessed February 18, 2026, [https://labs.aveni.ai/what-makes-memory-work-evaluating-long-term-memory-for-large-language-models/](https://labs.aveni.ai/what-makes-memory-work-evaluating-long-term-memory-for-large-language-models/)  
34. \[2601.07023\] CloneMem: Benchmarking Long-Term Memory for AI Clones \- arXiv.org, accessed February 18, 2026, [https://arxiv.org/abs/2601.07023](https://arxiv.org/abs/2601.07023)  
35. PersonaLens: A Benchmark for Personalization Evaluation in Conversational AI Assistants, accessed February 18, 2026, [https://arxiv.org/html/2506.09902v1](https://arxiv.org/html/2506.09902v1)  
36. arxiv.org, accessed February 18, 2026, [https://arxiv.org/html/2601.07023v1](https://arxiv.org/html/2601.07023v1)  
37. PersonaLens : A Benchmark for Personalization Evaluation in Conversational AI Assistants \- ACL Anthology, accessed February 18, 2026, [https://aclanthology.org/2025.findings-acl.927.pdf](https://aclanthology.org/2025.findings-acl.927.pdf)  
38. PrivacyBench: A Conversational Benchmark for Evaluating Privacy in Personalized AI \- arXiv, accessed February 18, 2026, [https://arxiv.org/html/2512.24848v1](https://arxiv.org/html/2512.24848v1)  
39. Benchmarking Hebbian learning rules for associative memory \- arXiv, accessed February 18, 2026, [https://arxiv.org/pdf/2401.00335](https://arxiv.org/pdf/2401.00335)  
40. \[2401.00335\] Benchmarking Hebbian learning rules for associative memory \- arXiv, accessed February 18, 2026, [https://arxiv.org/abs/2401.00335](https://arxiv.org/abs/2401.00335)  
41. \[PDF\] MemoryBank: Enhancing Large Language Models with Long-Term Memory, accessed February 18, 2026, [https://www.semanticscholar.org/paper/MemoryBank%3A-Enhancing-Large-Language-Models-with-Zhong-Guo/c3a59e1e405e7c28319e5a1c5b5241f9b340cf63](https://www.semanticscholar.org/paper/MemoryBank%3A-Enhancing-Large-Language-Models-with-Zhong-Guo/c3a59e1e405e7c28319e5a1c5b5241f9b340cf63)  
42. Modeling Memory Retention with Ebbinghaus's Forgetting Curve and Interpretable Machine Learning on Behavioral Factors \- ResearchGate, accessed February 18, 2026, [https://www.researchgate.net/publication/390933216\_Modeling\_Memory\_Retention\_with\_Ebbinghaus's\_Forgetting\_Curve\_and\_Interpretable\_Machine\_Learning\_on\_Behavioral\_Factors](https://www.researchgate.net/publication/390933216_Modeling_Memory_Retention_with_Ebbinghaus's_Forgetting_Curve_and_Interpretable_Machine_Learning_on_Behavioral_Factors)  
43. Human-like Forgetting Curves in Deep Neural Networks \- arXiv.org, accessed February 18, 2026, [https://arxiv.org/html/2506.12034v1](https://arxiv.org/html/2506.12034v1)  
44. Benchmarking Personality Inference in Large Language Models Using Real-World Conversations \- Journal of Psychiatry and Brain Science, accessed February 18, 2026, [https://jpbs.hapres.com/htmls/JPBS\_1832\_Detail.html](https://jpbs.hapres.com/htmls/JPBS_1832_Detail.html)  
45. Big5-Chat: Shaping LLM Personalities Through Training on Human-Grounded Data \- arXiv, accessed February 18, 2026, [https://arxiv.org/html/2410.16491v2](https://arxiv.org/html/2410.16491v2)  
46. Evaluating LLM Stability Under Self-Challenging Prompts | TELUS Digital, accessed February 18, 2026, [https://www.telusdigital.com/insights/data-and-ai/resource/evaluating-llm-stability-under-self-challenging-prompts](https://www.telusdigital.com/insights/data-and-ai/resource/evaluating-llm-stability-under-self-challenging-prompts)  
47. (PDF) PrivacyBench: A Conversational Benchmark for Evaluating Privacy in Personalized AI, accessed February 18, 2026, [https://www.researchgate.net/publication/399276331\_PrivacyBench\_A\_Conversational\_Benchmark\_for\_Evaluating\_Privacy\_in\_Personalized\_AI](https://www.researchgate.net/publication/399276331_PrivacyBench_A_Conversational_Benchmark_for_Evaluating_Privacy_in_Personalized_AI)  
48. Cognitive Workspace: Active Memory Management for LLMs An Empirical Study of Functional Infinite Context \- arXiv, accessed February 18, 2026, [https://arxiv.org/html/2508.13171v1](https://arxiv.org/html/2508.13171v1)  
49. Exploring Working Memory Capacity in LLMs: From Stressors to Human-Inspired Strategies \- ACL Anthology, accessed February 18, 2026, [https://aclanthology.org/2025.ijcnlp-long.93.pdf](https://aclanthology.org/2025.ijcnlp-long.93.pdf)  
50. Exploring Working Memory Capacity in LLMs: From Stressors to Human-Inspired Strategies, accessed February 18, 2026, [https://aclanthology.org/2025.ijcnlp-long.93/](https://aclanthology.org/2025.ijcnlp-long.93/)  
51. P50 vs P95 vs P99 Latency Explained: What Each Percentile Tells You \- OneUptime, accessed February 18, 2026, [https://oneuptime.com/blog/post/2025-09-15-p50-vs-p95-vs-p99-latency-percentiles/view](https://oneuptime.com/blog/post/2025-09-15-p50-vs-p95-vs-p99-latency-percentiles/view)  
52. Build and deploy scalable AI agents with NVIDIA NeMo, Amazon Bedrock AgentCore, and Strands Agents | Artificial Intelligence \- AWS, accessed February 18, 2026, [https://aws.amazon.com/blogs/machine-learning/build-and-deploy-scalable-ai-agents-with-nvidia-nemo-amazon-bedrock-agentcore-and-strands-agents/](https://aws.amazon.com/blogs/machine-learning/build-and-deploy-scalable-ai-agents-with-nvidia-nemo-amazon-bedrock-agentcore-and-strands-agents/)  
53. The Hidden Memory Architecture of LLMs | Microsoft Community Hub, accessed February 18, 2026, [https://techcommunity.microsoft.com/blog/educatordeveloperblog/the-hidden-memory-architecture-of-llms/4485367](https://techcommunity.microsoft.com/blog/educatordeveloperblog/the-hidden-memory-architecture-of-llms/4485367)  
54. letta-ai/letta-leaderboard: An LLM leaderboard for stateful ... \- GitHub, accessed February 18, 2026, [https://github.com/letta-ai/letta-leaderboard](https://github.com/letta-ai/letta-leaderboard)  
55. Long Term Memory \- Mem0/Zep/LangMem \- what made you choose it? : r/LangChain, accessed February 18, 2026, [https://www.reddit.com/r/LangChain/comments/1p0e4nk/long\_term\_memory\_mem0zeplangmem\_what\_made\_you/](https://www.reddit.com/r/LangChain/comments/1p0e4nk/long_term_memory_mem0zeplangmem_what_made_you/)  
56. (PDF) Designing AI-Agents with Personalities: A Psychometric Approach \- ResearchGate, accessed February 18, 2026, [https://www.researchgate.net/publication/385226256\_Designing\_AI-Agents\_with\_Personalities\_A\_Psychometric\_Approach](https://www.researchgate.net/publication/385226256_Designing_AI-Agents_with_Personalities_A_Psychometric_Approach)  
57. MCP-Atlas: A Large-Scale Benchmark for Tool-Use Competency with Real MCP Servers, accessed February 18, 2026, [https://arxiv.org/html/2602.00933v1](https://arxiv.org/html/2602.00933v1)  
58. MCP Atlas | SEAL by Scale AI, accessed February 18, 2026, [https://scale.com/leaderboard/mcp\_atlas](https://scale.com/leaderboard/mcp_atlas)  
59. Exploring Effective Testing Frameworks for AI Agents in Real-World Scenarios \- Maxim AI, accessed February 18, 2026, [https://www.getmaxim.ai/articles/exploring-effective-testing-frameworks-for-ai-agents-in-real-world-scenarios/](https://www.getmaxim.ai/articles/exploring-effective-testing-frameworks-for-ai-agents-in-real-world-scenarios/)  
60. AI agent evaluation: A practical framework for testing multi-step agents \- Articles \- Braintrust, accessed February 18, 2026, [https://www.braintrust.dev/articles/ai-agent-evaluation-framework](https://www.braintrust.dev/articles/ai-agent-evaluation-framework)  
61. Letta Evals: Evaluating Agents that Learn, accessed February 18, 2026, [https://www.letta.com/blog/letta-evals](https://www.letta.com/blog/letta-evals)  
62. Can Any Model Use Skills? Adding Skills to Context-Bench | Letta, accessed February 18, 2026, [https://www.letta.com/blog/context-bench-skills](https://www.letta.com/blog/context-bench-skills)  
63. \[2507.21474\] Hebbian Memory-Augmented Recurrent Networks: Engram Neurons in Deep Learning \- arXiv, accessed February 18, 2026, [https://arxiv.org/abs/2507.21474](https://arxiv.org/abs/2507.21474)  
64. Mem0 vs Zep vs LangMem vs MemoClaw: AI Agent Memory Comparison 2026 \- Dev.to, accessed February 18, 2026, [https://dev.to/anajuliabit/mem0-vs-zep-vs-langmem-vs-memoclaw-ai-agent-memory-comparison-2026-1l1k](https://dev.to/anajuliabit/mem0-vs-zep-vs-langmem-vs-memoclaw-ai-agent-memory-comparison-2026-1l1k)  
65. Is Letta the Best AI Agent Framework Right Now? An Honest Review \- Sider.AI, accessed February 18, 2026, [https://sider.ai/blog/ai-tools/is-letta-the-best-ai-agent-framework-right-now-an-honest-review](https://sider.ai/blog/ai-tools/is-letta-the-best-ai-agent-framework-right-now-an-honest-review)  
66. Mem0 vs Zep | Memory Layer Comparison \- Keywords AI, accessed February 18, 2026, [https://www.keywordsai.co/market-map/compare/mem0-vs-zep](https://www.keywordsai.co/market-map/compare/mem0-vs-zep)  
67. Which one is better for GraphRAG?: Cognee vs Graphiti vs Mem0 : r/Rag \- Reddit, accessed February 18, 2026, [https://www.reddit.com/r/Rag/comments/1qgbm8d/which\_one\_is\_better\_for\_graphrag\_cognee\_vs/](https://www.reddit.com/r/Rag/comments/1qgbm8d/which_one_is_better_for_graphrag_cognee_vs/)  
68. Can LLMs Correct Themselves? A Benchmark of Self-Correction in LLMs \- arXiv, accessed February 18, 2026, [https://arxiv.org/html/2510.16062v1](https://arxiv.org/html/2510.16062v1)  
69. \[2502.12110\] A-MEM: Agentic Memory for LLM Agents \- arXiv.org, accessed February 18, 2026, [https://arxiv.org/abs/2502.12110](https://arxiv.org/abs/2502.12110)  
70. A-Mem: Agentic Memory for LLM Agents \- arXiv, accessed February 18, 2026, [https://arxiv.org/html/2502.12110v11](https://arxiv.org/html/2502.12110v11)  
71. A-Mem: Agentic Memory for LLM Agents \- arXiv.org, accessed February 18, 2026, [https://arxiv.org/html/2502.12110v1](https://arxiv.org/html/2502.12110v1)  
72. AI Memory Systems Benchmark: Mem0 vs OpenAI vs LangMem 2025 \- Deepak Gupta, accessed February 18, 2026, [https://guptadeepak.com/the-ai-memory-wars-why-one-system-crushed-the-competition-and-its-not-openai/](https://guptadeepak.com/the-ai-memory-wars-why-one-system-crushed-the-competition-and-its-not-openai/)