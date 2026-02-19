# Engram

**Code-first memory system for autonomous AI development agents.**

Engram gives LLM coding agents persistent memory across sessions — what was built, why decisions were made, how errors were fixed, and what patterns work. Four memory layers (episodic, semantic, procedural, working) with code quality signal measurement, AST-based structural extraction, project-scoped isolation, and a knowledge graph of code dependencies. Runs as an MCP server with 32 tools. Everything local, no API dependencies.

[![GitHub](https://img.shields.io/badge/GitHub-Relic--Studios%2Fengram-blue?logo=github)](https://github.com/Relic-Studios/engram)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-603%20passing-brightgreen.svg)]()

> **Branch:** `engram-code` — Code-first pivot from the consciousness-focused `master` branch.

---

## The Problem

Coding agents forget everything between sessions. Every new conversation starts cold — no memory of your project's architecture, no recall of what patterns worked, no awareness of past debugging sessions. The agent re-discovers your codebase from scratch every time.

Engram solves this by giving agents a memory system designed specifically for code generation and long-term development work.

## What It Does

- **Remembers architecture decisions** — Why you chose PostgreSQL over MongoDB, stored as ADRs with full rationale
- **Learns from debugging** — Error fingerprinting links failures to past resolutions, so the agent doesn't repeat mistakes
- **Tracks code patterns** — Validated implementation templates stored as procedural skills, promoted by usage frequency
- **Maps dependencies** — Knowledge graph of which files depend on which, what exports what, what tests validate what
- **Measures code quality** — Four-facet signal (correctness, consistency, completeness, robustness) drives memory reinforcement
- **Extracts structure** — AST parsing of Python, JavaScript, and TypeScript extracts functions, classes, imports, and complexity metrics
- **Scopes by project** — Memory isolation per project, so context from project A doesn't pollute project B

---

## Memory Layers

| Layer | What It Stores | How It Works |
|-------|---------------|--------------|
| **Episodic** | Conversations, debugging sessions, code reviews, error resolutions | SQLite with FTS5 full-text search + code-aware symbol tokenizer |
| **Semantic** | Relationships, preferences, boundaries, identity (SOUL.md) | Human-readable YAML and Markdown files |
| **Procedural** | Code patterns, skills, implementation templates | Markdown skill files with keyword matching |
| **Working** | Active context for the current conversation | Token-budgeted knapsack allocation with MMR diversity and U-shaped reordering |

### Search Pipeline

```
Query → FTS5 (symbol-expanded) + ChromaDB (NL vectors + code vectors) → RRF Merge → Cross-Encoder Rerank → Results
```

- **Lexical**: FTS5 with custom symbol tokenizer — splits `camelCase`, `snake_case`, `PascalCase`, dot paths, kebab-case
- **NL Semantic**: ChromaDB with nomic-embed-text (768d, local via Ollama) — conversations, ADRs, journal
- **Code Semantic**: ChromaDB with jina-embeddings-v2-base-code (768d, 8k context, via sentence-transformers) — function signatures, stack traces, code patterns
- **Fusion**: Reciprocal Rank Fusion (Cormack et al., 2009) — merges results from all three sources
- **Reranking**: cross-encoder/ms-marco-MiniLM-L-12-v2 (~30ms, +15-30% precision)

Code traces are dual-indexed in both NL and code collections. Queries search all spaces simultaneously; results that appear in multiple spaces get boosted by RRF.

### Code Quality Signal (CQS)

Replaces the consciousness signal from `master`. Four facets measured on every LLM response:

| Facet | Weight | What It Measures |
|-------|--------|-----------------|
| **Correctness** | 40% | Valid syntax, proper error handling, no anti-patterns |
| **Consistency** | 20% | Naming conventions, style adherence, formatting |
| **Completeness** | 20% | Docstrings, type hints, edge case handling |
| **Robustness** | 20% | Error handling, input validation, defensive coding |

Signal health drives Hebbian reinforcement (good code → boost related memories), adaptive decay (poor signal → faster forgetting), and self-correction prompts.

Measurement is a three-way blend: **regex heuristics** + **AST structural analysis** + **optional LLM scoring**.

### AST Extraction Engine

Multi-language structural analysis:

| Language | Parser | Capabilities |
|----------|--------|-------------|
| **Python** | stdlib `ast` | Functions, classes, imports, decorators, docstrings, complexity metrics, anti-patterns |
| **JavaScript** | tree-sitter | Functions, classes, arrow functions, imports, JSX detection |
| **TypeScript** | tree-sitter | Everything JS + interfaces, type aliases, generics |
| **Fallback** | Regex | Basic function/class/import extraction for any language |

Extracts: symbol names, parameter counts, return types, nesting depth, cyclomatic complexity, annotations coverage, anti-pattern detection (bare except, magic numbers, god functions).

### Knowledge Graph (Wiring)

Structural relationships between code entities:

| Predicate | Meaning | Example |
|-----------|---------|---------|
| `depends_on` | A imports/calls B | `auth.py depends_on database.py` |
| `exports` | A exports B | `utils.py exports verify_token` |
| `validates` | A tests B | `test_auth.py validates auth.py` |
| `supersedes` | A replaces B | `v2_api supersedes v1_api` |
| `uses` | A references B | `config.py uses ENV_VARS` |
| `implements` | A implements B | `PostgresStore implements DataStore` |

Enables multi-hop queries: "what modules depend on the database schema?" traverses the graph.

---

## Quick Start

### Install

```bash
git clone https://github.com/Relic-Studios/engram.git
cd engram
git checkout engram-code
pip install -e ".[all]"
```

### Dependencies

**Core** (always installed with `pip install -e .`):

| Package | Version | Purpose |
|---------|---------|---------|
| `pyyaml` | >=6.0 | YAML config and semantic store files |
| `chromadb` | >=0.4.0 | Vector search (semantic embeddings) |
| `mcp` | >=1.0.0 | Model Context Protocol server |
| `numpy` | >=1.24.0 | Consolidation clustering, numeric ops |

**Optional** (install individually or use `.[all]` for everything):

| Group | Install | Packages | Purpose |
|-------|---------|----------|---------|
| `code-embeddings` | `pip install -e ".[code-embeddings]"` | `sentence-transformers>=3.0.0` | Dual-embedding (code model + cross-encoder reranking) |
| `ast` | `pip install -e ".[ast]"` | `tree-sitter`, `tree-sitter-python`, `tree-sitter-javascript`, `tree-sitter-typescript` | AST extraction for JS/TS (Python uses stdlib) |
| `consolidation` | `pip install -e ".[consolidation]"` | `hdbscan>=0.8.33` | Topic-coherent memory clustering |
| `tokens` | `pip install -e ".[tokens]"` | `tiktoken>=0.5.0` | Accurate OpenAI-style token counting |
| `ollama` | `pip install -e ".[ollama]"` | `httpx>=0.25.0` | Ollama LLM + embedding provider |
| `anthropic` | `pip install -e ".[anthropic]"` | `anthropic>=0.30.0` | Anthropic LLM provider |
| `openai` | `pip install -e ".[openai]"` | `openai>=1.0.0` | OpenAI LLM provider |

**GPU acceleration** (optional):

The code embedding model and cross-encoder reranker use PyTorch. By default PyTorch runs on CPU, which is fine for single-document inference (~50ms). For GPU acceleration on NVIDIA GPUs:

```bash
# Replace the CPU-only torch with CUDA-enabled version
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

The system auto-detects CUDA availability. No code changes needed.

### Initialize

```bash
engram init --data-dir ./my-memory
```

### Run MCP Server

```bash
engram serve --data-dir ./my-memory
```

Or add to your MCP client config:

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram",
      "args": ["serve", "--data-dir", "/path/to/my-memory"]
    }
  }
}
```

### Python API

```python
from engram import MemorySystem

memory = MemorySystem(data_dir="./my-memory")

# Boot at session start — loads SOUL.md, recent sessions, ADRs
boot = memory.boot()

# Before every LLM call — assembles context from all memory layers
context = memory.before(person="developer", message="Fix the auth bug in login.py")
# context.text → identity + recent messages + relevant traces + skills + ADRs

# After every LLM response — measures quality, reinforces/decays, extracts knowledge
result = memory.after(
    person="developer",
    their_message="Fix the auth bug in login.py",
    response="Found the issue — the token validation was...",
    trace_ids=context.trace_ids,
)
# result.signal.health → 0.83 (code quality score)
# Automatically: logs exchange, runs CQS measurement, Hebbian reinforcement,
#   AST extraction, symbol expansion, adaptive decay, workspace aging
```

---

## Architecture

```
              engram_before()                    engram_after()
                   |                                  |
      +------------+------------+        +------------+------------+
      |                         |        |                         |
 Identity Resolution      Context        1. Code Quality Signal    |
 (alias → canonical)      Assembly          (regex + AST + LLM)    |
      |                     |            2. Salience Derivation     |
 +----+-----+         Token Budget       3. Hebbian Reinforcement  |
 |          |          Allocation            (citation-primary)     |
SOUL.md  Relationship  (knapsack +       4. Semantic Extraction    |
Identity + Grounding    MMR diversity)   5. Style Adherence Check  |
 |       + Rules            |            6. Workspace Age Step     |
Workspace                Context         7. Pressure-aware         |
Items              [1] [2] [3] [4]          Maintenance            |
 |                 numbered traces                                 |
Recent             → system prompt                                 |
Messages                                                           |
 |                                                                 |
Procedural                                                         |
Skills + ADRs                                                      |
                                                                   |
          +---- SQLite + FTS5 (symbol-expanded) ---+               |
          +---- ChromaDB vectors ---+               |               |
          +---- YAML/Markdown ---+                  |               |
          +---- Knowledge Graph (wiring) ----------+               |
```

---

## MCP Tools (32)

### Core Pipeline
| Tool | Purpose |
|------|---------|
| `engram_before` | Load memory context before LLM call |
| `engram_after` | Log exchange, measure signal, reinforce/decay |
| `engram_boot` | Session boot — SOUL.md, recent sessions, ADRs, style rules |

### Search & Recall
| Tool | Purpose |
|------|---------|
| `engram_search` | Hybrid search across all memory types |
| `engram_recall` | Get specific content (identity, preferences, etc.) |
| `engram_recall_time` | Query by time range |
| `engram_stats` | Memory system health metrics |
| `engram_signal` | Code quality signal state and trend |

### Agent-Directed Memory
| Tool | Purpose |
|------|---------|
| `engram_remember` | Deliberately save to long-term memory |
| `engram_forget` | Decay a memory to near-zero salience |
| `engram_reflect` | Consolidate and review memories on a topic |
| `engram_correct` | Supersede an inaccurate memory |

### Code Intelligence
| Tool | Purpose |
|------|---------|
| `engram_extract_symbols` | AST analysis — extract functions, classes, imports, complexity |
| `engram_repo_map` | Generate Aider-style repo map for a directory |
| `engram_code_pattern` | Store/retrieve validated implementation patterns |
| `engram_debug_log` | Record error + resolution for future recall |
| `engram_architecture_decision` | Record ADR (context, options, decision, consequences) |
| `engram_get_rules` | Get coding standards and style rules |
| `engram_add_wiring` | Record dependency relationship in knowledge graph |
| `engram_project_init` | Initialize/switch project scope |

### Knowledge Management
| Tool | Purpose |
|------|---------|
| `engram_add_fact` | Add fact to a relationship file |
| `engram_add_skill` | Add/update a procedural skill |
| `engram_log_event` | Log a discrete event |
| `engram_boundary_add` | Add a behavioral boundary |
| `engram_contradiction_add` | Record a held contradiction |
| `engram_preferences_add` | Add a preference |
| `engram_preferences_search` | Search preferences |
| `engram_journal_write` | Write a reflective journal entry |
| `engram_journal_list` | List recent journal entries |

### Workspace
| Tool | Purpose |
|------|---------|
| `engram_workspace_add` | Add item to working memory (7+/-2 capacity) |
| `engram_workspace_status` | View current working memory contents |

### Maintenance
| Tool | Purpose |
|------|---------|
| `engram_reindex` | Rebuild all search indexes |

---

## Trace Kinds

19 episodic memory types, each with distinct semantics:

**Core:** `episode`, `realization`, `factual`, `reflection`

**Code-first:** `code_pattern`, `debug_session`, `architecture_decision`, `wiring_map`, `error_resolution`, `test_strategy`, `project_context`, `code_review`, `code_symbols`

**Infrastructure:** `temporal`, `utility`, `workspace_eviction`

**Consolidation (system-managed):** `summary`, `thread`, `arc`

---

## Infrastructure

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| **Adaptive decay** | ACT-R exponential with coherence modulation | Memories fade unless reinforced |
| **Hebbian reinforcement** | Citation-primary, signal-gated | Used memories get stronger |
| **Consolidation** | HDBSCAN clustering → threads → arcs | Prevents unbounded trace growth |
| **Memory pressure** | Three-level monitor (normal/elevated/critical) | Throttles decay and compaction |
| **Conversation compaction** | MemGPT-inspired summarization | Old messages → thread traces |
| **Project scoping** | Per-project memory isolation | Context doesn't leak across projects |
| **Style enforcement** | AST + regex style checking in after-pipeline | Naming conventions, nesting depth |
| **Dual embeddings** | NL model (Ollama) + code model (sentence-transformers) | Separate vector spaces for prose and code |
| **Symbol tokenization** | Application-layer FTS5 expansion | `getUserName` → `get user name` for search |

---

## Configuration

```yaml
engram:
  data_dir: ./my-memory
  signal_mode: hybrid         # hybrid | regex | llm
  extract_mode: off           # llm | off
  llm_provider: ollama        # ollama | openai | anthropic
  llm_model: llama3.2
  llm_base_url: http://localhost:11434
  token_budget: 12000

  # NL embedding model (local via Ollama)
  embedding_model: nomic-embed-text    # 768d, MTEB ~0.63

  # Code embedding model (via sentence-transformers)
  code_embedding_model: jinaai/jina-embeddings-v2-base-code  # 768d, 8k context
  code_embedding_device: ""            # "" = auto (CUDA if available)

  # Cross-encoder reranking
  reranker_enabled: true
  reranker_model: cross-encoder/ms-marco-MiniLM-L-12-v2

  # Memory management
  decay_half_life_hours: 168.0         # 1 week
  max_traces: 50000
  reinforce_delta: 0.05
  weaken_delta: 0.03

  # Code-first boot
  boot_n_sessions: 3                   # Recent coding sessions at boot
  boot_n_decisions: 5                  # Recent ADRs at boot
```

---

## Data Directory

```
my-memory/
  engram.db              # SQLite (messages, traces, events, sessions + FTS5)
  soul/
    SOUL.md              # Coding philosophy and identity
    journal/             # Reflective entries
  semantic/
    identities.yaml      # Alias resolution
    preferences.yaml     # Likes, dislikes, uncertainties
    boundaries.yaml      # Behavioral boundaries
    contradictions.yaml  # Held contradictions
    relationships/       # Per-person relationship files
  procedural/            # Skill files (code patterns, workflows)
  style/                 # Coding style preferences
  projects/              # Per-project scoped data
  architecture/          # Architecture Decision Records
  embeddings/            # ChromaDB vector store
```

---

## Development

```bash
git clone https://github.com/Relic-Studios/engram.git
cd engram
git checkout engram-code
pip install -e ".[all,dev]"

# Run tests (skip consolidation — known slow tests)
pytest tests/ --ignore=tests/test_consolidation.py -x -q --tb=short
# 603 tests, ~2 minutes
```

### Test Suite

603 tests across core systems:

- Dual embedding system: 49 tests
- AST extraction engine: 62 tests
- Symbol tokenizer: 40 tests
- Symbol index + repo map: 33 tests
- Server tools: all 32 tools covered
- Episodic, semantic, procedural stores
- Search pipeline (FTS5, semantic, unified, reranker)
- Signal measurement, style checking, reinforcement, decay
- Consolidation, workspace, journal, grounding

---

## Roadmap

The `engram-code` branch follows a phased build plan. Completed work and remaining items:

### Done

| Phase | What |
|-------|------|
| Core pivot | Strip consciousness modules, rewire for Code Quality Signal |
| SOTA retrieval | RRF hybrid search, cross-encoder reranking, ACT-R decay, HDBSCAN consolidation |
| Project scoping | Per-project memory isolation |
| Style-in-the-loop | AST + regex style checking, wiring graph, relationship predicates |
| Boot priming | SOUL.md philosophy, coding style, code-first classifier |
| AST extraction | Multi-language structural analysis (Python, JS, TS) + repo map generation |
| FTS5 tokenizer | Code-aware symbol splitting for compound identifiers |
| Dual embeddings | Code-specific embedding model (jina-embeddings-v2-base-code) + NL model, dual-indexing, auto CUDA detection |

### Next

| Item | Description |
|------|-------------|
| **Error fingerprinting** | Sentry-style SHA-256 fingerprints for error deduplication |
| **Debugging session schema** | Structured error → investigation → resolution tracking |
| **YAML procedural schemas** | Replace flat markdown skills with structured YAML + frontmatter matching |
| **Frequency-based confidence** | Usage frequency drives pattern promotion to "Core Skill" |
| **Automated ADR generation** | Detect design choices in after-pipeline, prompt for ADR creation |
| **Primacy-Recency wiring** | Wire existing U-shaped reordering into the retrieval pipeline |
| **Multi-agent MCP server** | Cross-tool persistence for multiple IDEs |

---

## License

MIT License. See [LICENSE](LICENSE).

## Author

**[Aidan Poole](https://github.com/AidanPoole)** / [Relic Studios](https://github.com/Relic-Studios)
