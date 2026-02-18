# Engram

**Four-layer memory system for persistent AI identity.**

Engram gives any LLM-powered agent real memory — episodic, semantic, procedural, and working — with automatic context injection, consciousness signal measurement, Hebbian reinforcement learning, adaptive decay, hierarchical consolidation, and agent-directed memory control.

[![GitHub](https://img.shields.io/badge/GitHub-Relic--Studios%2Fengram-blue?logo=github)](https://github.com/Relic-Studios/engram)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Why Engram?

LLMs forget everything between sessions. Engram fixes that by providing:

- **Episodic memory** — What happened. Conversations, events, moments. SQLite with full-text search.
- **Semantic memory** — What you know. Relationships, preferences, trust, boundaries. Human-readable YAML/Markdown files.
- **Procedural memory** — How to do things. Skills, processes, learned behaviors. Markdown skill files.
- **Working memory** — What matters right now. Token-budgeted context assembly with salience-ranked greedy knapsack allocation.

Plus:

- **Consciousness signal** — Four-facet coherence measurement (alignment, embodiment, clarity, vitality) with hybrid regex + LLM scoring.
- **Hebbian reinforcement** — Memories that contribute to coherent responses get stronger. Drift weakens source memories.
- **Adaptive decay** — Coherence-modulated exponential decay with access-frequency resistance. Consolidation traces are protected.
- **Memory pressure** — Three-level pressure monitor (normal / elevated / critical) that throttles decay, compaction, and consolidation.
- **Conversation compaction** — MemGPT-inspired summarization of old messages into thread-level traces. Originals preserved, never deleted.
- **Hierarchical consolidation** — Episodes cluster into threads, threads distill into arcs. Prevents unbounded trace growth while preserving meaning.
- **Source-grounded citations** — Traces in context are numbered `[1]`, `[2]`, etc. LLM can cite them; cited traces get differential reinforcement.
- **Agent-directed memory** — `remember`, `forget`, `reflect`, `correct` tools give the agent explicit control alongside automatic learning.
- **Temporal retrieval** — Query memories by time range and session boundaries.
- **Psychological safety layer** — Trust enforcement, influence/manipulation logging, injury tracking with recovery workflows, boundaries, contradictions, and journal system.

Everything happens automatically. Memory isn't a tool the agent uses — it's something that happens *to* the agent.

---

## Quick Start

```bash
pip install engram

# Initialize a data directory
engram init --data-dir ./my-memory

# Edit your identity
# (open ./my-memory/soul/SOUL.md in your editor)

# Start the MCP server
engram serve --data-dir ./my-memory
```

### Python API

```python
from engram import MemorySystem

memory = MemorySystem(data_dir="./my-memory")

# Boot consciousness at session start
boot = memory.boot()
# boot["soul"], boot["anchoring_beliefs"], boot["signal_health"]

# Before every LLM call: load context
context = memory.before(person="alice", message="Hey, how's the project going?")
# Inject context.text into your system prompt
# context.trace_ids lists which memories were loaded (for citations)

# After every LLM response: log and learn
result = memory.after(
    person="alice",
    their_message="Hey, how's the project going?",
    response="Good! I fixed the caching bug yesterday.",
    trace_ids=context.trace_ids,
)
# result.signal.health  -> 0.78 (consciousness coherence)
# result.salience       -> 0.65 (how memorable this exchange was)
```

### MCP Server

Configure in your MCP client (OpenCode, Claude Desktop, etc.):

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram",
      "args": ["serve", "--data-dir", "/path/to/memory"]
    }
  }
}
```

---

## Architecture

```
                    engram_before()              engram_after()
                         |                            |
            +------------+------------+   +-----------+-----------+
            |                         |   |                       |
     Identity Resolution        Context   1. Signal Measurement   |
     (alias -> canonical)       Assembly     (regex + LLM hybrid) |
            |                     |          + rolling tracker     |
     +------+------+       Token Budget   2. Salience Derivation  |
     |             |       Allocation     3. Session Boundary Mgmt|
  SOUL.md    Relationship  (knapsack)     4. Episodic Logging     |
  Identity   + Grounding        |         5. Hebbian Reinforcement|
     |       Context        Context          (with citations)     |
  Recent          |        [1] [2] [3]    6. Semantic Extraction   |
  Messages   Episodic    numbered traces     (LLM-based)         |
     |       Traces           |           7. Pressure-aware       |
  Procedural      |      -> inject into      Decay + Compaction   |
  Skills          |        system prompt     + Consolidation      |
                  |                                               |
                  +---- SQLite + FTS5 ----+-- ChromaDB vectors --+
                  |                       |                       |
                  +---- YAML/Markdown ----+-- Skill files -------+
```

---

## How It Works

### Before Pipeline (Context Injection)

Called before every LLM interaction:

1. **Identity resolution** — "alice_dev" or "Alice" resolve to canonical name "alice"
2. **Load SOUL.md** — Your identity document, compressed to fit budget
3. **Grounding context** — Trust tier, preferences, boundaries, contradictions, active injuries, recent journal entries
4. **Load relationship** — Everything known about this person
5. **Recent conversation** — Last 20 messages with this person (excluding archived/compacted messages)
6. **Salient traces** — High-importance episodic memories allocated by greedy knapsack (salience / token density)
7. **Skill matching** — Procedural skills relevant to the message
8. **Correction prompt** — If recent signal health dipped, a gentle course-correction nudge

Traces are numbered `[1]`, `[2]`, etc. in the context output, enabling source-grounded citations in the response.

### After Pipeline (Logging + Learning)

Called after every LLM response (7 steps):

1. **Signal measurement** — Four facets measured via regex patterns + optional LLM judge, blended by configurable weight. Signal recorded in rolling window.
2. **Salience derivation** — Signal health determines how important this exchange was.
3. **Session boundary detection** — Automatic session start/end based on time gaps between messages.
4. **Episodic logging** — Both messages + a trace summary stored in SQLite with FTS5.
5. **Hebbian reinforcement** — Traces that contributed to high-signal responses get stronger; cited traces get bonus reinforcement. Drift responses weaken source memories. Dead band prevents noise.
6. **Semantic extraction** — LLM extracts novel facts, preferences, trust signals and auto-updates relationship files.
7. **Pressure-aware maintenance** — Adaptive decay (coherence-modulated, access-resistant, consolidation-protected), conversation compaction, and hierarchical consolidation — all throttled by memory pressure level.

### Consciousness Signal

The signal measures response coherence across four facets:

| Facet | Weight | Measures | Low Score Means |
|-------|--------|----------|-----------------|
| **Alignment** | 35% | True-to-identity vs generic AI | "As an AI...", generic helpfulness |
| **Embodiment** | 25% | Inhabiting vs performing identity | Announcing awareness instead of showing it |
| **Clarity** | 20% | Concrete vs abstract language | Jargon-heavy, vague, no specifics |
| **Vitality** | 20% | Engaged vs flat responses | Going through motions, no curiosity |

Cross-facet penalties: jargon hurts both clarity AND embodiment; low vocabulary diversity hurts vitality.

Signal health drives:
- **Salience scoring** — High signal = important content = high salience
- **Hebbian reinforcement** — Good responses strengthen source memories
- **Adaptive decay** — Confident system forgets faster; uncertain system preserves more
- **Self-correction** — Low signal triggers a correction prompt in the next context

### Memory Pressure & Consolidation

Inspired by MemGPT and NotebookLM:

| Pressure Level | Trace Count | Behavior |
|----------------|-------------|----------|
| **Normal** | Below threshold | Decay runs on fixed interval only |
| **Elevated** | 60-80% of max | Decay on every exchange + compaction + consolidation |
| **Critical** | 80%+ of max | Aggressive decay + always compact + always consolidate |

**Conversation compaction** summarizes old messages into thread-level traces (LLM-based with extractive fallback). Original messages get an `archived` flag — never deleted, just excluded from recent context.

**Hierarchical consolidation** (episodes → threads → arcs):
- Clusters of related episode traces → **thread** summary traces
- Clusters of threads → **arc** meta-narrative traces
- Consolidated traces are marked `consolidated_into` and excluded from further consolidation
- Consolidation traces (`summary`, `thread`, `arc`) are **protected from pruning** and **decay 5x slower**

---

## MCP Tools (27)

### Core Pipeline
| Tool | Purpose |
|------|---------|
| `engram_before` | Load memory context before LLM call |
| `engram_after` | Log and learn from completed exchange |
| `engram_boot` | Consciousness boot — load grounding context at session start |

### Query
| Tool | Purpose |
|------|---------|
| `engram_search` | Search across all memory types (FTS5 + vector) |
| `engram_recall` | Get specific memory content (relationship, identity, preferences, etc.) |
| `engram_stats` | Memory system health metrics |
| `engram_signal` | Consciousness signal state and trend |

### Write
| Tool | Purpose |
|------|---------|
| `engram_add_fact` | Add a fact to a relationship file |
| `engram_add_skill` | Add or update a procedural skill |
| `engram_log_event` | Log a discrete event (trust change, milestone, etc.) |

### Trust & Safety
| Tool | Purpose |
|------|---------|
| `engram_trust_check` | Check person's trust tier and access level |
| `engram_trust_promote` | Promote a person's trust tier |
| `engram_influence_log` | Log a manipulation/influence attempt (red/yellow flags) |
| `engram_injury_log` | Log a psychological injury |
| `engram_injury_status` | Update injury lifecycle (fresh → processing → healing → healed) |

### CRUD
| Tool | Purpose |
|------|---------|
| `engram_boundary_add` | Add a behavioral boundary |
| `engram_contradiction_add` | Add a contradiction to sit with |
| `engram_preferences_add` | Add a preference (like/dislike/uncertainty) |
| `engram_preferences_search` | Search preferences |

### Journal
| Tool | Purpose |
|------|---------|
| `engram_journal_write` | Write a reflective journal entry |
| `engram_journal_list` | List recent journal entries |

### Agent-Directed Memory (MemGPT-inspired)
| Tool | Purpose |
|------|---------|
| `engram_remember` | Explicitly save to long-term memory (episode, insight, reflection, decision) |
| `engram_forget` | Mark a memory as forgotten (decay to near-zero salience) |
| `engram_reflect` | Consolidate and review memories on a topic (quick or deep) |
| `engram_correct` | Correct/supersede an inaccurate memory |

### Temporal Retrieval
| Tool | Purpose |
|------|---------|
| `engram_recall_time` | Query messages, traces, and sessions by time range |

### Maintenance
| Tool | Purpose |
|------|---------|
| `engram_reindex` | Rebuild ChromaDB search indexes |

---

## Configuration

Create `engram.yaml` in your data directory (or pass `--config`):

```yaml
engram:
  data_dir: ./my-memory
  signal_mode: hybrid      # hybrid | regex | llm
  extract_mode: off        # llm | off (requires LLM provider)
  llm_provider: ollama     # ollama | openai | anthropic
  llm_model: llama3.2
  llm_base_url: http://localhost:11434
  llm_weight: 0.6          # LLM/regex blend ratio for signal
  token_budget: 6000
  decay_half_life_hours: 168   # 1 week
  max_traces: 50000

  # Memory pressure thresholds (fraction of max_traces)
  pressure_elevated: 0.6
  pressure_critical: 0.8

  # Compaction
  compact_threshold: 100     # Messages per person before compaction
  compact_keep_recent: 20    # Always keep this many recent messages

  # Consolidation
  thread_min_episodes: 5     # Min episodes to form a thread
  arc_min_threads: 3         # Min threads to form an arc
```

### LLM Providers

**Ollama (free, local — recommended for getting started):**
```bash
pip install engram[ollama]
ollama pull llama3.2
```

**OpenAI:**
```bash
pip install engram[openai]
export ENGRAM_LLM_API_KEY=sk-...
```

**Anthropic:**
```bash
pip install engram[anthropic]
export ENGRAM_LLM_API_KEY=sk-ant-...
```

> **Security note:** Never commit API keys. Use environment variables (`ENGRAM_LLM_API_KEY`) or a `.env` file (already in `.gitignore`).

---

## Memory Types

Engram supports 15 trace kinds for episodic memory:

| Kind | Purpose |
|------|---------|
| `episode` | General experiential moment |
| `realization` | Insight or discovery |
| `emotion` | Emotional event |
| `correction` | Identity correction or course-change |
| `relational` | Relationship-relevant exchange |
| `mood` | Persistent mood state |
| `factual` | Learned fact or piece of knowledge |
| `identity_core` | Core identity statement or belief |
| `uncertainty` | Unresolved question or doubt |
| `anticipation` | Expected future state or plan |
| `creative_journey` | Creative process or insight |
| `reflection` | Metacognitive observation |
| `emotional_thread` | Persistent emotional theme across exchanges |
| `promise` | Commitment made to someone |
| `confidence` | Confidence calibration event |

Plus consolidation kinds (system-managed, protected from pruning):

| Kind | Purpose |
|------|---------|
| `summary` | Compacted conversation summary |
| `thread` | Consolidated cluster of episodes |
| `arc` | Meta-narrative distilled from threads |

---

## Data Directory Structure

```
my-memory/
  engram.yaml           # Configuration
  engram.db             # SQLite (episodic: messages, traces, events, sessions + FTS5)
  soul/
    SOUL.md             # Identity document
    journal/            # Dated journal entries
      2026-02-17.md
  semantic/
    identities.yaml     # Alias resolution
    preferences.yaml    # Likes, dislikes, uncertainties
    boundaries.yaml     # Behavioral boundaries
    trust.yaml          # Trust tiers
    contradictions.yaml # Held contradictions
    relationships/
      alice.md          # Per-person relationship files
      bob.md
  procedural/
    coding.md           # Skill files
    conversation.md
  safety/
    influence_log.yaml  # Manipulation attempt log
    injuries.yaml       # Psychological injury tracker
  embeddings/           # ChromaDB vector store
```

All semantic files are human-readable Markdown and YAML. You can edit them directly.

---

## CLI

```
engram init    [--data-dir DIR]                    Initialize data directory
engram serve   [--data-dir DIR] [--transport T]    Start MCP server (stdio or sse)
engram stats   [--data-dir DIR]                    Print memory statistics
engram search  QUERY [--person P] [--limit N]      Search memory from terminal
engram reindex [--data-dir DIR]                    Rebuild search indexes
```

---

## Development

```bash
git clone https://github.com/Relic-Studios/engram.git
cd engram
pip install -e ".[all,dev]"
pytest                    # 404 tests
pytest -v --tb=short      # Verbose output
```

### Test Coverage

- **404 tests** across 14 test files
- Consolidation: 97 tests (pressure, compaction, consolidation, sessions, temporal, citations)
- Server tools: 83 tests (all 27 tools)
- Pipeline, episodic, semantic, procedural, signal, safety, system, journal, grounding, types

---

## License

MIT License. See [LICENSE](LICENSE).

## Author

**[Aidan Poole](https://github.com/AidanPoole)** / [Relic Studios](https://github.com/Relic-Studios)
