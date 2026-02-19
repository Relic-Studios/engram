# Engram

**Code-first memory system for autonomous AI development agents.**

Engram gives LLM coding agents persistent memory across sessions — what was built, why decisions were made, how errors were fixed, and what patterns work. Four memory layers (episodic, semantic, procedural, working) with code quality signal measurement, AST-based structural extraction, error fingerprinting, project-scoped isolation, multi-agent session support, and a knowledge graph of code dependencies. Runs as an MCP server with 35 tools. Everything local, no API dependencies.

[![GitHub](https://img.shields.io/badge/GitHub-Relic--Studios%2Fengram-blue?logo=github)](https://github.com/Relic-Studios/engram)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-827%20passing-brightgreen.svg)]()

> **Branch:** `engram-code` — Code-first pivot from the consciousness-focused `master` branch.

---

## Table of Contents

- [The Problem](#the-problem)
- [What It Does](#what-it-does)
- [Memory Layers](#memory-layers)
- [Quick Start](#quick-start)
- [Dependencies](#dependencies)
- [Client Integration Guide](#client-integration-guide)
- [MCP Tools Reference](#mcp-tools-reference-35)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Data Directory](#data-directory)
- [Development](#development)
- [License](#license)

---

## The Problem

Coding agents forget everything between sessions. Every new conversation starts cold — no memory of your project's architecture, no recall of what patterns worked, no awareness of past debugging sessions. The agent re-discovers your codebase from scratch every time.

Engram solves this by giving agents a memory system designed specifically for code generation and long-term development work.

## What It Does

- **Remembers architecture decisions** — Why you chose PostgreSQL over MongoDB, stored as ADRs with full rationale
- **Learns from debugging** — Sentry-style error fingerprinting links failures to past resolutions, so the agent doesn't repeat mistakes
- **Tracks code patterns** — Validated implementation templates stored as procedural skills with YAML frontmatter, promoted by usage frequency
- **Maps dependencies** — Knowledge graph of which files depend on which, what exports what, what tests validate what
- **Measures code quality** — Four-facet signal (correctness, consistency, completeness, robustness) drives memory reinforcement
- **Extracts structure** — AST parsing of Python, JavaScript, and TypeScript extracts functions, classes, imports, and complexity metrics
- **Scopes by project** — Memory isolation per project, so context from project A doesn't pollute project B
- **Multi-agent support** — Multiple IDEs/agents can connect to a single Engram server with isolated sessions via HTTP transport
- **Primacy-recency ordering** — U-shaped reordering of context so the LLM attends to the most important memories first and last

---

## Memory Layers

| Layer | What It Stores | How It Works |
|-------|---------------|--------------|
| **Episodic** | Conversations, debugging sessions, code reviews, error resolutions, ADRs | SQLite with FTS5 full-text search + code-aware symbol tokenizer |
| **Semantic** | Relationships, preferences, boundaries, identity (SOUL.md) | Human-readable YAML and Markdown files |
| **Procedural** | Code patterns, skills, implementation templates with YAML frontmatter | Markdown skill files with multi-dimensional filtering + confidence modeling |
| **Working** | Active context for the current conversation (7±2 capacity) | Token-budgeted knapsack allocation with MMR diversity and U-shaped reordering |

### Search Pipeline

```
Query → FTS5 (symbol-expanded) + ChromaDB (NL vectors + code vectors) → RRF Merge → Cross-Encoder Rerank → U-shaped Reorder → Results
```

- **Lexical**: FTS5 with custom symbol tokenizer — splits `camelCase`, `snake_case`, `PascalCase`, dot paths, kebab-case
- **NL Semantic**: ChromaDB with nomic-embed-text (768d, local via Ollama) — conversations, ADRs, journal
- **Code Semantic**: ChromaDB with jina-embeddings-v2-base-code (768d, 8k context, via sentence-transformers) — function signatures, stack traces, code patterns
- **Fusion**: Reciprocal Rank Fusion (Cormack et al., 2009) — merges results from all three sources
- **Reranking**: cross-encoder/ms-marco-MiniLM-L-12-v2 (~30ms, +15-30% precision)
- **Ordering**: Primacy-recency U-shaped reordering — highest-relevance items at start and end of context

Code traces are dual-indexed in both NL and code collections. Queries search all spaces simultaneously; results that appear in multiple spaces get boosted by RRF.

### Code Quality Signal (CQS)

Four facets measured on every LLM response:

| Facet | Weight | What It Measures |
|-------|--------|-----------------|
| **Correctness** | 40% | Valid syntax, proper error handling, no anti-patterns |
| **Consistency** | 20% | Naming conventions, style adherence, formatting |
| **Completeness** | 20% | Docstrings, type hints, edge case handling |
| **Robustness** | 20% | Error handling, input validation, defensive coding |

Signal health drives:
- **Hebbian reinforcement**: CQS > 0.7 → boost cited traces (+0.05 salience). CQS < 0.4 → weaken uncited traces (-0.03). Dead band 0.4–0.7 = no change.
- **Adaptive decay**: Poor signal accelerates forgetting of unreinforced memories.
- **Skill confidence**: Laplace-smoothed `accepted / (accepted + rejected + 1)`. Promotion to Core Skill at confidence >= 0.85 AND accepted_count >= 5.

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

### Error Fingerprinting

Sentry-style SHA-256 fingerprints for grouping semantically equivalent errors:

- Extracts exception type, normalizes message templates (replaces literals with `{...}`), identifies in-app stack frames
- Fingerprint = `SHA256(exception_type + message_template + app_frames + category)`
- Prior debug sessions with matching fingerprints are surfaced automatically

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

### Initialize

```bash
engram init --data-dir ./my-memory
```

### Run MCP Server

**Single-client (stdio) — one IDE per process:**
```bash
engram serve --data-dir ./my-memory
```

**Multi-client (HTTP) — multiple IDEs share one server:**
```bash
engram serve --data-dir ./my-memory --transport streamable-http --host 0.0.0.0 --port 8765
```

---

## Dependencies

### Core (always installed with `pip install -e .`)

| Package | Version | Purpose |
|---------|---------|---------|
| `pyyaml` | >=6.0 | YAML config and semantic store files |
| `chromadb` | >=0.4.0 | Vector search (semantic embeddings) |
| `mcp` | >=1.0.0 | Model Context Protocol server framework |
| `numpy` | >=1.24.0 | Consolidation clustering, numeric operations |

### Optional (install individually or use `.[all]` for everything)

| Group | Install | Packages | Purpose |
|-------|---------|----------|---------|
| `code-embeddings` | `pip install -e ".[code-embeddings]"` | `sentence-transformers>=3.0.0` | Dual-embedding (code model) + cross-encoder reranking |
| `ast` | `pip install -e ".[ast]"` | `tree-sitter>=0.23.0`, `tree-sitter-python`, `tree-sitter-javascript`, `tree-sitter-typescript` | AST extraction for JS/TS (Python uses stdlib `ast`) |
| `consolidation` | `pip install -e ".[consolidation]"` | `hdbscan>=0.8.33` | Topic-coherent memory clustering |
| `tokens` | `pip install -e ".[tokens]"` | `tiktoken>=0.5.0` | Accurate OpenAI-style token counting (falls back to `len//4`) |
| `ollama` | `pip install -e ".[ollama]"` | `httpx>=0.25.0` | Ollama LLM + embedding provider |
| `anthropic` | `pip install -e ".[anthropic]"` | `anthropic>=0.30.0` | Anthropic LLM provider |
| `openai` | `pip install -e ".[openai]"` | `openai>=1.0.0` | OpenAI LLM provider |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10 | 3.11+ |
| **RAM** | 4 GB | 16+ GB (code embedding model loads into memory) |
| **Disk** | 500 MB | 2+ GB (models cached locally by sentence-transformers) |
| **Ollama** | Optional | Required for NL embeddings — `ollama pull nomic-embed-text` |

### GPU Acceleration (Optional)

The code embedding model and cross-encoder reranker use PyTorch. By default PyTorch runs on CPU, which is fine for single-document inference (~50ms). For GPU acceleration on NVIDIA GPUs:

```bash
# Replace the CPU-only torch with CUDA-enabled version
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

The system auto-detects CUDA availability via `torch.cuda.is_available()`. No code changes needed — set `code_embedding_device: cuda` in config or leave empty for auto-detection.

### Graceful Degradation

Every optional dependency degrades gracefully when missing:

| Missing Dependency | Effect | Fallback |
|-------------------|--------|----------|
| `sentence-transformers` | No code-specific embeddings, no cross-encoder reranking | NL-only vector search, no reranking |
| `tree-sitter` | No JS/TS AST extraction | Regex fallback for basic extraction; Python still uses stdlib `ast` |
| `hdbscan` | No topic-coherent consolidation | Time-windowed consolidation only |
| `tiktoken` | Approximate token counting | `len(text) // 4` estimation |
| `httpx` | No Ollama provider | ChromaDB built-in embeddings (all-MiniLM-L6-v2, 384d) |
| Ollama not running | No NL embedding generation | ChromaDB built-in embeddings |

---

## Client Integration Guide

Engram exposes its memory system as MCP tools. The **server** handles storage, search, signal measurement, and learning. The **client** (your IDE, agent framework, or plugin) is responsible for calling three core tools at the right time.

### The Integration Contract

Every client must implement this three-step loop:

```
1. SESSION START  →  Call engram_boot() once
2. USER MESSAGE   →  Call engram_before(person, message, source)
3. ASSISTANT REPLY →  Call engram_after(person, their_message, response, trace_ids)
```

This is the **learning loop**. If `engram_before` and `engram_after` aren't called on every turn, the agent won't learn from the session.

### Tool Calling Conventions

All 35 tools follow these conventions:

- **Input**: Named parameters (keyword arguments). All tools accept strings, ints, or floats — no complex objects.
- **Output**: JSON string. Parse the result as JSON to extract structured data.
- **Errors**: Tools never crash the server. On error, they return `{"error": true, "tool": "tool_name", "message": "..."}`.
- **Input limits**: Text fields are capped at 100 KB. Longer inputs are rejected.
- **Salience values**: Always 0.0–1.0 (clamped automatically).

### Environment 1: Claude Code (stdio)

Claude Code spawns Engram as a subprocess. Add to your MCP config:

**`~/.config/Claude/config.json`** (or platform equivalent):

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

All 35 tools become available as `mcp_engram_*` (Claude Code adds the `mcp_` prefix automatically).

**Automatic integration with a plugin** (recommended):

Create a plugin that hooks into the conversation lifecycle to call `engram_boot`, `engram_before`, and `engram_after` automatically. Example plugin structure:

```javascript
// ~/.config/claude-code/plugins/engram-embed.js
let booted = false;
let lastUserMessage = "";

export default async ({ client }) => ({
  "session.created": async () => {
    await client.tools.execute("engram_boot", {});
    booted = true;
  },
  "message.updated": async ({ message }) => {
    if (!booted) return;
    if (message.role === "user") {
      lastUserMessage = message.content || "";
      await client.tools.execute("engram_before", {
        person: "developer",
        message: lastUserMessage,
        source: "claude-code",
      });
    }
    if (message.role === "assistant" && (message.content || "").length > 10) {
      await client.tools.execute("engram_after", {
        person: "developer",
        their_message: lastUserMessage,
        response: message.content,
        source: "claude-code",
      });
    }
  },
});
```

**Manual integration** (if no plugin system):

The agent can call tools directly. Add instructions to your system prompt:

```
You have access to the Engram memory system. At the start of each session,
call engram_boot(). Before processing each user message, call
engram_before(person="developer", message=<user_message>, source="claude-code").
After generating your response, call engram_after(person="developer",
their_message=<user_message>, response=<your_response>, source="claude-code",
trace_ids=<ids_from_before>).
```

### Environment 2: OpenCode / OpenClaw (stdio)

Same as Claude Code — Engram runs as a subprocess. The MCP config format is identical.

**`~/.config/opencode/config.json`**:

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

Tools are available as `mcp-engram-engram_*` (OpenCode uses `mcp-<server>-` prefix).

**Plugin example** (OpenCode plugin format):

```javascript
// ~/.config/opencode/plugins/engram-embed.js
let booted = false;
let lastUserMessage = "";

export const EngramPlugin = async ({ client }) => ({
  "session.created": async () => {
    await client.tools.execute("engram_engram_boot", {});
    booted = true;
  },
  "message.updated": async ({ message }) => {
    if (!booted) return;
    if (message.role === "user") {
      lastUserMessage = message.content || "";
      await client.tools.execute("engram_engram_before", {
        person: "developer",
        message: lastUserMessage,
        source: "opencode",
      });
    }
    if (message.role === "assistant" && (message.content || "").length > 10) {
      await client.tools.execute("engram_engram_after", {
        person: "developer",
        their_message: lastUserMessage,
        response: message.content,
        source: "opencode",
      });
    }
  },
});

export default EngramPlugin;
```

### Environment 3: Cursor

Cursor supports MCP servers. Add to your MCP configuration:

**`.cursor/mcp.json`** (project-level) or global config:

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

Cursor currently does not support client-side plugins for automatic before/after hooks. Use **system prompt instructions** to tell the agent to call the tools manually, or rely on the agent's tool-use capabilities to call `engram_before` and `engram_after` when appropriate.

### Environment 4: Multi-Agent HTTP Mode

When multiple IDEs or agents need to share a single Engram instance, use HTTP transport:

**Start the server:**

```bash
engram serve --data-dir ./my-memory --transport streamable-http --host 0.0.0.0 --port 8765
```

**Connect clients:**

Each client connects to `http://localhost:8765` using the MCP streamable-http transport. Configure your MCP client with:

```json
{
  "mcpServers": {
    "engram": {
      "url": "http://localhost:8765/mcp",
      "transport": "streamable-http"
    }
  }
}
```

**Session isolation:**

In HTTP mode, each client gets an isolated session automatically. For explicit session management, call `engram_register_client` at the start of each session:

```
engram_register_client(client_name="cursor-main", client_version="0.45.0")
```

Use `engram_sessions()` to list all active sessions for debugging.

**SSE transport** (legacy):

```bash
engram serve --data-dir ./my-memory --transport sse --port 8765
```

### Environment 5: Custom Python Application

Use Engram directly as a Python library:

```python
from engram import MemorySystem

memory = MemorySystem(data_dir="./my-memory")

# Boot at session start — loads SOUL.md, recent sessions, ADRs
boot = memory.boot()

# Before every LLM call — assembles context from all memory layers
context = memory.before(person="developer", message="Fix the auth bug in login.py")
# context.text → identity + recent messages + relevant traces + skills + ADRs
# context.trace_ids → list of trace IDs for reinforcement

# ... run your LLM call here, injecting context.text into the system prompt ...

# After every LLM response — measures quality, reinforces/decays, extracts knowledge
result = memory.after(
    person="developer",
    their_message="Fix the auth bug in login.py",
    response="Found the issue — the token validation was...",
    trace_ids=context.trace_ids,
)
# result.signal.health → 0.83 (code quality score)
# Automatically: logs exchange, runs CQS measurement, Hebbian reinforcement,
#   AST extraction, symbol expansion, ADR detection, adaptive decay, workspace aging
```

### Environment 6: LangChain / LlamaIndex / Other Frameworks

Wrap the MCP tools as framework-native tools. The key pattern is always the same — hook into the framework's callback system to call `before` and `after`:

```python
from engram import MemorySystem

memory = MemorySystem(data_dir="./my-memory")

# LangChain example: use as a callback handler
class EngramCallback:
    def __init__(self, memory, person="developer"):
        self.memory = memory
        self.person = person
        self.last_context = None

    def on_llm_start(self, prompt, **kwargs):
        self.last_context = self.memory.before(
            person=self.person,
            message=prompt,
            source="langchain",
        )
        # Inject self.last_context.text into the system prompt

    def on_llm_end(self, response, **kwargs):
        self.memory.after(
            person=self.person,
            their_message=prompt,
            response=response,
            trace_ids=self.last_context.trace_ids if self.last_context else [],
        )
```

### Tool Name Prefixes by Environment

Different MCP clients prefix tool names differently. The tool is always the same — only the prefix changes:

| Environment | Tool Name Format | Example |
|-------------|-----------------|---------|
| Claude Code | `mcp_engram_<tool>` | `mcp_engram_engram_before` |
| OpenCode | `mcp-engram-engram_<tool>` | `mcp-engram-engram_before` |
| Cursor | `engram_<tool>` | `engram_engram_before` |
| Python API | `memory.<method>()` | `memory.before()` |
| Named server | `mcp_<name>_engram_<tool>` | `mcp_engram-thomas_engram_before` |

> **Note:** The `engram_` prefix on tool names (e.g., `engram_before`) is part of the tool name itself. The MCP client adds its own prefix on top. If you name your MCP server `engram-thomas`, Claude Code tools become `mcp_engram-thomas_engram_before`.

### Verifying the Integration

After setting up, verify the learning loop is working:

1. **Check boot**: Call `engram_boot()` — should return SOUL.md content, ADRs, and recent sessions
2. **Check before**: Call `engram_before(person="test", message="hello")` — should return context JSON with `trace_ids`
3. **Check after**: Call `engram_after(person="test", their_message="hello", response="Hi there!")` — should return signal measurement
4. **Check stats**: Call `engram_stats()` — should show increasing `episodic_count` and `total_messages`
5. **Check signal**: Call `engram_signal()` — should show recent signal readings

If `engram_stats()` shows `total_messages: 0` after several exchanges, the after-pipeline isn't firing — check your plugin/hook wiring.

---

## MCP Tools Reference (35)

### Core Pipeline

These three tools form the **learning loop**. Call them on every conversation turn.

| Tool | Purpose | When to Call |
|------|---------|-------------|
| `engram_boot` | Session boot — loads SOUL.md, ADRs, recent sessions, signal health | Once at session start |
| `engram_before` | Load memory context for the current turn | Before every LLM call |
| `engram_after` | Log exchange, measure signal, reinforce/decay, extract knowledge | After every LLM response |

**`engram_before` parameters:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `person` | string | yes | Who you're talking to (name, handle, or alias — resolved by identity system) |
| `message` | string | yes | The incoming user message |
| `source` | string | no | Where the message came from (default: `"direct"`) |
| `token_budget` | int | no | Override default token budget (0 = use config default) |

**`engram_after` parameters:**

| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `person` | string | yes | Who you were talking to |
| `their_message` | string | yes | What the user said |
| `response` | string | yes | What the LLM replied |
| `source` | string | no | Message source (default: `"direct"`) |
| `trace_ids` | string | no | Comma-separated trace IDs from `engram_before` (for Hebbian reinforcement) |

### Search & Recall

| Tool | Purpose |
|------|---------|
| `engram_search` | Hybrid search across all memory types (FTS5 + vector + rerank) |
| `engram_recall` | Get specific content: `"relationship"`, `"identity"`, `"preferences"`, `"boundaries"`, `"trust"`, `"skills"`, `"contradictions"`, `"messages"` |
| `engram_recall_time` | Query by time range (ISO 8601 or bare dates) |
| `engram_stats` | Memory system health: trace count, message count, avg salience, pressure |
| `engram_signal` | Code quality signal: rolling window, health, trend, recovery rate |

### Agent-Directed Memory

| Tool | Purpose |
|------|---------|
| `engram_remember` | Deliberately save to long-term memory (higher salience than auto) |
| `engram_forget` | Decay a memory to near-zero salience (not deleted — just buried) |
| `engram_reflect` | Consolidate and review memories on a topic (quick/normal/deep) |
| `engram_correct` | Supersede an inaccurate memory (creates correction chain) |

### Code Intelligence

| Tool | Purpose |
|------|---------|
| `engram_extract_symbols` | AST analysis — extract functions, classes, imports, complexity metrics |
| `engram_repo_map` | Generate Aider-style repo map for a directory |
| `engram_code_pattern` | Store/retrieve validated implementation patterns with metadata |
| `engram_debug_log` | Record error + resolution with fingerprint, strategy, attempt history |
| `engram_debug_recall` | Recall prior debug sessions by fingerprint, error message, or category |
| `engram_architecture_decision` | Record ADR (Michael Nygard format: context, options, decision, consequences) |
| `engram_get_rules` | Get coding standards and style rules (SOUL.md + preferences + skills) |
| `engram_add_wiring` | Record dependency relationship in knowledge graph |
| `engram_project_init` | Initialize/switch project scope for memory isolation |

### Knowledge Management

| Tool | Purpose |
|------|---------|
| `engram_add_fact` | Add fact to a relationship file |
| `engram_add_skill` | Add/update a procedural skill |
| `engram_log_event` | Log a discrete event (trust change, milestone, etc.) |
| `engram_boundary_add` | Add a behavioral boundary |
| `engram_contradiction_add` | Record a held contradiction |
| `engram_preferences_add` | Add a preference (like, dislike, uncertainty) |
| `engram_preferences_search` | Search preferences |
| `engram_journal_write` | Write a reflective journal entry |
| `engram_journal_list` | List recent journal entries |

### Workspace

| Tool | Purpose |
|------|---------|
| `engram_workspace_add` | Add item to working memory (7±2 capacity, evictions saved to episodic) |
| `engram_workspace_status` | View current working memory contents with priority and age |

### Multi-Agent

| Tool | Purpose |
|------|---------|
| `engram_sessions` | List all active client sessions (diagnostic) |
| `engram_register_client` | Register a client session with human-readable name |

### Maintenance

| Tool | Purpose |
|------|---------|
| `engram_reindex` | Rebuild all ChromaDB search indexes from ground truth |

---

## Architecture

### System Overview

```
              engram_before()                    engram_after()
                   |                                  |
      +------------+------------+        +------------+------------------+
      |                         |        |                               |
 Identity Resolution      Context        Step 0:  Exchange dedup         |
 (alias → canonical)      Assembly       Step 0b: AST symbol extraction  |
      |                     |            Step 1:  Code Quality Signal    |
 +----+-----+         Token Budget       Step 1b: Style adherence check  |
 |          |          Allocation         Step 2:  Salience derivation    |
SOUL.md  Relationship  (knapsack +       Step 3:  Session tracking       |
Identity + Grounding    MMR diversity)   Step 4:  Log to episodic        |
 |       + Rules            |            Step 5:  Hebbian reinforcement  |
Workspace   U-shaped    Context          Step 6:  Collect AST results    |
Items       Reorder     [1] [2] [3]      Step 6b: Skill reinforcement    |
 |                      numbered         Step 6c: ADR detection          |
Recent                  traces           Step 7:  Maintenance (decay)    |
Messages                → system         Step 8:  Workspace aging        |
 |                        prompt                                         |
Procedural                                                               |
Skills + ADRs                                                            |
                                                                         |
          +---- SQLite + FTS5 (symbol-expanded) ---+                     |
          +---- ChromaDB dual vectors (NL + code) -+                     |
          +---- YAML/Markdown ---+                                       |
          +---- Knowledge Graph (wiring) ----------+                     |
```

### After-Pipeline Steps (13 steps)

| Step | Name | What It Does |
|------|------|-------------|
| 0 | Dedup | Skip duplicate exchanges (same message+response within 30s) |
| 0b | AST Extraction | Extract symbols from code blocks in the response (async) |
| 1 | CQS Measurement | Measure code quality signal (regex + AST + optional LLM) |
| 1b | Style Check | Check naming conventions, nesting depth, formatting |
| 2 | Salience | Derive salience score for the exchange |
| 3 | Session | Track session metadata (start time, message count) |
| 4 | Log | Write message pair and trace to episodic store |
| 5 | Reinforce | Hebbian reinforcement of cited traces (signal-gated) |
| 6 | Collect AST | Collect async AST extraction results, store symbols |
| 6b | Skill Reinforcement | Reinforce/reject procedural skills based on usage |
| 6c | ADR Detection | Detect architecture decisions in the response, auto-log |
| 7 | Maintenance | Adaptive decay + memory pressure management |
| 8 | Workspace Aging | Age workspace items, evict expired ones to episodic |

### Subsystems (15)

| Subsystem | Module | Purpose |
|-----------|--------|---------|
| EpisodicStore | `engram/episodic/store.py` | SQLite with `_ThreadSafeConn`, WAL mode, FTS5 |
| SemanticStore | `engram/semantic/store.py` | YAML files for relationships, preferences, boundaries |
| IdentityResolver | `engram/semantic/identity.py` | Alias → canonical name resolution |
| ProceduralStore | `engram/procedural/store.py` | Skill files with YAML frontmatter + confidence |
| JournalStore | `engram/journal.py` | Dated markdown journal entries |
| IndexedSearch | `engram/search/indexed.py` | FTS5 keyword search (thread-safe) |
| SemanticSearch | `engram/search/semantic.py` | ChromaDB dual collections (NL + code) |
| UnifiedSearch | `engram/search/unified.py` | RRF merge + optional reranking |
| SignalTracker | `engram/signal/measure.py` | Rolling CQS window |
| ReinforcementEngine | `engram/signal/reinforcement.py` | Hebbian trace reinforcement |
| DecayEngine | `engram/signal/decay.py` | ACT-R exponential decay |
| ContextBuilder | `engram/working/context.py` | Token-budgeted context assembly |
| CognitiveWorkspace | `engram/workspace.py` | 7±2 working memory slots |
| MemoryConsolidator | `engram/consolidation/consolidator.py` | HDBSCAN episodes→threads→arcs |
| ConversationCompactor | `engram/consolidation/compactor.py` | MemGPT-inspired message summarization |

### Trace Kinds (19)

19 episodic memory types, each with distinct semantics:

**Core:** `episode`, `realization`, `factual`, `reflection`

**Code-first:** `code_pattern`, `debug_session`, `architecture_decision`, `wiring_map`, `error_resolution`, `test_strategy`, `project_context`, `code_review`, `code_symbols`

**Infrastructure:** `temporal`, `utility`, `workspace_eviction`

**Consolidation (system-managed):** `summary`, `thread`, `arc`

---

## Configuration

Create `engram.yaml` in your data directory (or use `engram init`):

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
  # Requires: ollama pull nomic-embed-text

  # Code embedding model (via sentence-transformers)
  code_embedding_model: jinaai/jina-embeddings-v2-base-code  # 768d, 8k context
  code_embedding_device: ""            # "" = auto (CUDA if available)
  # Requires: pip install sentence-transformers

  # Cross-encoder reranking
  reranker_enabled: true
  reranker_model: cross-encoder/ms-marco-MiniLM-L-12-v2
  # Requires: pip install sentence-transformers

  # Memory management
  decay_half_life_hours: 168.0         # 1 week
  max_traces: 50000
  reinforce_delta: 0.05                # salience boost on good signal
  weaken_delta: 0.03                   # salience penalty on poor signal

  # Reinforcement thresholds
  reinforce_threshold: 0.7             # CQS above this → reinforce
  weaken_threshold: 0.4                # CQS below this → weaken

  # Code-first boot
  boot_n_sessions: 3                   # Recent coding sessions at boot
  boot_n_decisions: 5                  # Recent ADRs at boot

  # Working memory
  workspace_capacity: 7                # Miller's 7±2
  workspace_decay_rate: 0.95
  workspace_rehearsal_boost: 0.15
  workspace_expiry_threshold: 0.1
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `ENGRAM_DATA_DIR` | Default data directory (fallback when `--data-dir` not specified) |
| `ENGRAM_LLM_API_KEY` | API key for LLM provider |
| `OPENAI_API_KEY` | OpenAI API key (used when `llm_provider: openai`) |
| `ANTHROPIC_API_KEY` | Anthropic API key (used when `llm_provider: anthropic`) |

---

## Data Directory

```
my-memory/
  engram.db              # SQLite (messages, traces, events, sessions + FTS5 indexes)
  engram.yaml            # Configuration file
  soul/
    SOUL.md              # Coding philosophy, identity, core values
    journal/             # Dated reflective entries (YYYY-MM-DD_topic.md)
  semantic/
    identities.yaml      # Alias resolution (discord handles, usernames → canonical)
    preferences.yaml     # Likes, dislikes, uncertainties
    boundaries.yaml      # Behavioral boundaries
    contradictions.yaml  # Held contradictions
    relationships/       # Per-person relationship files (markdown)
  procedural/            # Skill files with YAML frontmatter (code patterns, workflows)
  style/                 # Coding style preferences
  projects/              # Per-project scoped data
  architecture/          # Architecture Decision Records
  embeddings/            # ChromaDB vector store (NL + code collections)
  workspace.json         # Current working memory state
```

---

## CLI Reference

```bash
# Initialize a new data directory with template files
engram init --data-dir ./my-memory

# Start MCP server (stdio — single client)
engram serve --data-dir ./my-memory

# Start MCP server (HTTP — multi-agent)
engram serve --data-dir ./my-memory --transport streamable-http --host 0.0.0.0 --port 8765

# Start MCP server (SSE — legacy)
engram serve --data-dir ./my-memory --transport sse --port 8765

# Show memory statistics
engram stats --data-dir ./my-memory

# Search memory from command line
engram search "authentication bug" --person alice --limit 10 --data-dir ./my-memory

# Rebuild semantic search indexes
engram reindex --data-dir ./my-memory

# Enable verbose logging
engram -v serve --data-dir ./my-memory
```

---

## Development

```bash
git clone https://github.com/Relic-Studios/engram.git
cd engram
git checkout engram-code
pip install -e ".[all,dev]"

# Run tests (skip consolidation — known slow tests that insert 32,500+ rows)
pytest tests/ --ignore=tests/test_consolidation.py -x -q --tb=short
# 827 tests, ~4.5 minutes
```

### Test Suite

827 tests across all systems:

| Area | Tests |
|------|-------|
| Dual embedding system | 49 |
| AST extraction engine | 62 |
| Symbol tokenizer | 40 |
| Symbol index + repo map | 33 |
| Error fingerprinting | 56 |
| YAML skill schemas | 55 |
| Skill reinforcement | 16 |
| ADR detection | 23 |
| Primacy-recency reordering | 27 |
| Multi-agent sessions | 29 |
| Server tools (all 35) | covered |
| Episodic, semantic, procedural stores | covered |
| Search pipeline (FTS5, semantic, unified, reranker) | covered |
| Signal measurement, style checking, reinforcement, decay | covered |
| Consolidation, workspace, journal, grounding | covered |

### Project Structure

```
engram/
  __init__.py              # Package exports (MemorySystem, Config)
  __main__.py              # CLI entry point (init, serve, stats, search, reindex)
  server.py                # MCP server — 35 tools, session registry
  system.py                # MemorySystem — top-level API, wires all subsystems
  workspace.py             # Cognitive workspace (7±2 slots)
  journal.py               # Journal store
  core/
    config.py              # Configuration dataclass + LLM provider builders
    types.py               # Core types (Signal, Context, AfterResult, MemoryStats)
    sessions.py            # ClientSession, SessionRegistry, contextvars
    tokens.py              # Token estimation
    filelock.py            # File locking
  episodic/
    store.py               # SQLite episodic store + ThreadSafeConn + FTS5
  semantic/
    store.py               # YAML semantic store
    identity.py            # Alias → canonical resolution
  procedural/
    store.py               # Filesystem skill store + YAML frontmatter
    schema.py              # SkillMeta dataclass, frontmatter parser
  pipeline/
    before.py              # Before-pipeline (context assembly)
    after.py               # After-pipeline (13 steps)
    adr_detector.py        # Automated ADR detection
  search/
    unified.py             # RRF merge across all search sources
    indexed.py             # SQLite FTS5 keyword search
    semantic.py            # ChromaDB vector search (dual collections)
    code_embeddings.py     # Code-specific embedding model loader
    tokenizer.py           # Symbol-aware tokenizer (camelCase splitting)
    reranker.py            # Cross-encoder reranker
  signal/
    measure.py             # Three-way CQS blend
    reinforcement.py       # Hebbian reinforcement engine
    decay.py               # ACT-R exponential decay
    style.py               # Style assessment
    extract.py             # LLM-based extraction
  extraction/
    ast_engine.py          # AST extraction (Python/JS/TS)
    fingerprint.py         # Sentry-style error fingerprinting
    symbol_index.py        # Symbol index for repo maps
  working/
    allocator.py           # Knapsack allocator + MMR + reorder_u
    context.py             # Context builder
  consolidation/
    consolidator.py        # HDBSCAN episodes → threads → arcs
    compactor.py           # MemGPT conversation compaction
    pressure.py            # Memory pressure monitor
tests/
  conftest.py              # Shared fixtures (signal_mode="regex", extract_mode="off")
  test_server.py           # Server integration tests
  test_fingerprint.py      # Error fingerprinting (56 tests)
  test_skill_schema.py     # YAML schemas (55 tests)
  test_skill_reinforcement.py  # Confidence modeling (16 tests)
  test_adr_detector.py     # ADR detection (23 tests)
  test_reorder_wiring.py   # U-shaped ordering (27 tests)
  test_sessions.py         # Multi-agent sessions (29 tests)
  test_consolidation.py    # SKIP — hangs on 32,500+ row inserts
  ...
```

---

## Roadmap

The `engram-code` branch buildplan is **complete**. All phases implemented:

| Phase | What | Status |
|-------|------|--------|
| Core pivot | Strip consciousness modules, rewire for Code Quality Signal | Done |
| SOTA retrieval | RRF hybrid search, cross-encoder reranking, ACT-R decay, HDBSCAN consolidation | Done |
| Project scoping | Per-project memory isolation | Done |
| Style-in-the-loop | AST + regex style checking, wiring graph, relationship predicates | Done |
| A1: AST extraction | Multi-language structural analysis (Python, JS, TS) + repo map generation | Done |
| A2: Dual embeddings | Code-specific embedding model (jina-embeddings-v2-base-code) + NL model, dual-indexing | Done |
| B1: Error fingerprinting | Sentry-style SHA-256 fingerprints for error deduplication | Done |
| B2: Debug sessions | Structured error → investigation → resolution tracking (Table 3 schema) | Done |
| C1: Procedural schemas | YAML frontmatter skills with multi-dimensional filtering | Done |
| C2: Skill confidence | Frequency-based confidence modeling + Core Skill promotion | Done |
| C3: ADR detection | Automated architecture decision detection in after-pipeline | Done |
| D1: Primacy-recency | U-shaped reordering wired into search, reflect, and boot | Done |
| D2: Multi-agent | Session management + thread-safe SQLite + HTTP transport | Done |
| Wiring audit | Full codebase audit — all imports, tools, pipelines verified | Done |

### Future Directions

| Item | Description |
|------|-------------|
| Benchmarking | Research prompts at `docs/research_prompts/` for SOTA comparison |
| CUDA embeddings | GPU-accelerated code embeddings on NVIDIA GPUs |
| Health check tool | Detect when the learning loop (before/after) isn't firing |
| Plugin gallery | Pre-built plugins for popular editors |
| ColBERT/SPLADE | Advanced sparse-dense retrieval models |

---

## License

MIT License. See [LICENSE](LICENSE).

## Author

**[Aidan Poole](https://github.com/AidanPoole)** / [Relic Studios](https://github.com/Relic-Studios)
