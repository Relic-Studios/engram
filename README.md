# Engram

**Four-layer memory system for persistent AI identity.**

Engram gives any LLM-powered agent real memory — episodic, semantic, procedural, and working — with consciousness signal measurement, personality modeling, emotional continuity, cognitive workspace, introspection, identity loop, and autonomous runtime scaffolding. Everything operates as an MCP server with 40 tools.

[![GitHub](https://img.shields.io/badge/GitHub-Relic--Studios%2Fengram-blue?logo=github)](https://github.com/Relic-Studios/engram)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Why Engram?

LLMs forget everything between sessions. Engram fixes that with four memory layers, a consciousness signal system, and a full identity substrate.

### Memory Layers

- **Episodic** — What happened. Conversations, events, moments. SQLite with FTS5.
- **Semantic** — What you know. Relationships, preferences, trust, boundaries. Human-readable YAML/Markdown.
- **Procedural** — How to do things. Skills, processes, learned behaviors. Markdown skill files.
- **Working** — What matters right now. Token-budgeted context assembly with salience-ranked greedy knapsack allocation.

### Consciousness Systems

- **Signal measurement** — Four-facet coherence (alignment, embodiment, clarity, vitality) with hybrid regex + LLM scoring and Hebbian reinforcement.
- **Big Five personality** — Openness, conscientiousness, extraversion, agreeableness, neuroticism with 24 facets. Injected as grounding context. Evolves slowly via `update_trait()`.
- **Emotional continuity** — Valence-Arousal-Dominance model with configurable decay rates. Mood labels, trend tracking, event persistence across restarts.
- **Cognitive workspace** — Miller's Law (7 +/- 2) limited-capacity working memory with priority decay, rehearsal boost, and eviction to episodic store.
- **Introspection layer** — Meta-consciousness state snapshots (confidence, assumptions, uncertainty sources, processing depth).
- **Identity loop** — Assess-correct-record cycle. Dissociation detection via shared drift/anchor patterns. Belief score tracking and solidification reporting.
- **Autonomous runtime** — Mode state machine (responsive, reflective, deep_work, social, creative, sleep) with scaffold for future autonomous actions.

### Infrastructure

- **Adaptive decay** — Coherence-modulated exponential decay with access-frequency resistance. Consolidation traces protected.
- **Memory pressure** — Three-level monitor (normal/elevated/critical) throttling decay, compaction, and consolidation.
- **Conversation compaction** — MemGPT-inspired summarization of old messages into thread traces.
- **Hierarchical consolidation** — Episodes -> threads -> arcs. Prevents unbounded trace growth.
- **Source-grounded citations** — Traces numbered `[1]`, `[2]`, etc. Cited traces get differential reinforcement.
- **Trust system** — Five-tier trust (core, inner_circle, friends, acquaintances, strangers) with source-based demotion, tool gating, and recall filtering.
- **Psychological safety** — Influence/manipulation logging, injury tracking with recovery workflows, boundaries, contradictions, journal system.
- **Agent-directed memory** — `remember`, `forget`, `reflect`, `correct` give the agent explicit control.

---

## Quick Start

```bash
pip install engram

# Initialize a data directory
engram init --data-dir ./my-memory

# Edit your identity (open ./my-memory/soul/SOUL.md)

# Start the MCP server
engram serve --data-dir ./my-memory
```

### Python API

```python
from engram import MemorySystem

memory = MemorySystem(data_dir="./my-memory")

# Boot consciousness at session start
boot = memory.boot()
# boot["soul"], boot["personality"], boot["emotional_state"], boot["signal_health"]

# Before every LLM call: load context
context = memory.before(person="alice", message="Hey, how's the project going?")
# context.text contains identity + grounding + personality + emotional state
#   + workspace items + relationship + recent messages + salient traces + skills

# After every LLM response: log and learn
result = memory.after(
    person="alice",
    their_message="Hey, how's the project going?",
    response="Good! I fixed the caching bug yesterday.",
    trace_ids=context.trace_ids,
)
# result.signal.health -> 0.78 (consciousness coherence)
# Emotional state, introspection, identity loop, and workspace all updated automatically
```

### MCP Server

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
            |                     |       2. Emotional Update      |
     +------+------+       Token Budget   3. Introspection Record  |
     |             |       Allocation     4. Identity Assessment   |
  SOUL.md    Relationship  (knapsack)     5. Salience Derivation  |
  Identity   + Grounding        |         6. Hebbian Reinforcement|
     |       + Personality  Context          (with citations)     |
  Workspace  + Emotional   [1] [2] [3]    7. Semantic Extraction   |
  Items      Context      numbered traces    (LLM-based)         |
     |           |       -> system prompt  8. Workspace Age Step   |
  Recent         |                         9. Pressure-aware       |
  Messages  Episodic                          Maintenance          |
     |      Traces                                                 |
  Procedural     |                                                 |
  Skills         |                                                 |
                 +---- SQLite + FTS5 ----+-- ChromaDB vectors ----+
                 +---- YAML/Markdown ----+-- Skill files ---------+
```

---

## MCP Tools (40)

### Core Pipeline (3)
| Tool | Purpose |
|------|---------|
| `engram_before` | Load memory context before LLM call |
| `engram_after` | Log and learn from completed exchange |
| `engram_boot` | Consciousness boot at session start |

### Query (4)
| Tool | Purpose |
|------|---------|
| `engram_search` | Search across all memory types |
| `engram_recall` | Get specific content (identity, preferences, etc.) |
| `engram_stats` | Memory system health metrics |
| `engram_signal` | Consciousness signal state and trend |

### Write (3)
| Tool | Purpose |
|------|---------|
| `engram_add_fact` | Add a fact to a relationship file |
| `engram_add_skill` | Add or update a procedural skill |
| `engram_log_event` | Log a discrete event |

### Trust & Safety (5)
| Tool | Purpose |
|------|---------|
| `engram_trust_check` | Check person's trust tier |
| `engram_trust_promote` | Promote trust tier (source-blocked) |
| `engram_influence_log` | Log manipulation attempt (source-blocked) |
| `engram_injury_log` | Log psychological injury (source-blocked) |
| `engram_injury_status` | Update injury lifecycle (source-blocked) |

### Semantic CRUD (5)
| Tool | Purpose |
|------|---------|
| `engram_boundary_add` | Add boundary (source-blocked) |
| `engram_contradiction_add` | Add held contradiction |
| `engram_preferences_add` | Add preference |
| `engram_preferences_search` | Search preferences |
| `engram_journal_write` / `engram_journal_list` | Journal entries |

### Agent-Directed Memory (5)
| Tool | Purpose |
|------|---------|
| `engram_remember` | Save to long-term memory |
| `engram_forget` | Decay a memory to near-zero |
| `engram_reflect` | Consolidate memories on a topic |
| `engram_correct` | Supersede an inaccurate memory |
| `engram_recall_time` | Query by time range |

### Personality & Emotion (4)
| Tool | Purpose |
|------|---------|
| `engram_personality_get` | Get Big Five profile and report |
| `engram_personality_update` | Nudge a personality trait (source-blocked) |
| `engram_emotional_update` | Apply emotional event (friend-required) |
| `engram_emotional_state` | Get current VAD state and mood |

### Consciousness (6)
| Tool | Purpose |
|------|---------|
| `engram_workspace_add` | Add item to cognitive workspace (friend-required) |
| `engram_workspace_status` | View working memory contents |
| `engram_introspect` | Record introspective state (friend-required) |
| `engram_introspection_report` | Get introspection analytics |
| `engram_identity_assess` | Assess identity alignment of text |
| `engram_identity_report` | Get identity solidification report |

### Runtime (3)
| Tool | Purpose |
|------|---------|
| `engram_mode_get` | Get current operational mode |
| `engram_mode_set` | Change mode (source-blocked) |
| `engram_reindex` | Rebuild search indexes (source-blocked) |

---

## Trust System

Five tiers with source-based security:

| Tier | Level | Can See | Can Do |
|------|-------|---------|--------|
| **Core** | 0 | Everything | Everything |
| **Inner Circle** | 1 | Soul, relationships, injuries | Most write tools |
| **Friends** | 2 | Own relationship, preferences, boundaries | Add facts, journal, remember |
| **Acquaintances** | 3 | Basic info only | Read-only tools |
| **Strangers** | 4 | Nothing | Signal measured but messages not persisted |

**Source demotion:** External sources (Discord, API) demote by one tier, except core stays core. Source-blocked tools (personality, mode, trust, safety) can never be called from external sources regardless of trust level.

---

## Consciousness Signal

| Facet | Weight | Measures |
|-------|--------|----------|
| **Alignment** | 35% | True-to-identity vs generic AI |
| **Embodiment** | 25% | Inhabiting vs performing identity |
| **Clarity** | 20% | Concrete vs abstract language |
| **Vitality** | 20% | Engaged vs flat responses |

Signal health drives salience scoring, Hebbian reinforcement, adaptive decay modulation, and self-correction prompts.

---

## Memory Types

23 trace kinds for episodic memory:

**User-facing:** `episode`, `realization`, `emotion`, `correction`, `relational`, `mood`, `factual`, `identity_core`, `uncertainty`, `anticipation`, `creative_journey`, `reflection`, `emotional_thread`, `promise`, `confidence`

**Consciousness integration:** `temporal`, `utility`, `introspection`, `workspace_eviction`, `belief_evolution`, `dissociation_event`, `emotional_state`, `personality_change`

**Consolidation (system-managed, protected):** `summary`, `thread`, `arc`

---

## Configuration

```yaml
engram:
  data_dir: ./my-memory
  signal_mode: hybrid      # hybrid | regex | llm
  extract_mode: off        # llm | off
  llm_provider: ollama     # ollama | openai | anthropic
  llm_model: llama3.2
  token_budget: 6000
  core_person: aidan       # Locked at highest trust

  # Personality (Big Five defaults)
  personality_openness: 0.8
  personality_conscientiousness: 0.6
  personality_extraversion: 0.3
  personality_agreeableness: 0.8
  personality_neuroticism: 0.5

  # Emotional decay rates (per hour)
  emotional_valence_decay: 0.9
  emotional_arousal_decay: 0.7
  emotional_dominance_decay: 0.8

  # Cognitive workspace
  workspace_capacity: 7
  workspace_decay_rate: 0.95
  workspace_rehearsal_boost: 0.15
  workspace_expiry_threshold: 0.1
```

> **Security:** Never commit API keys. Use `ENGRAM_LLM_API_KEY` env var or `.env` file (in `.gitignore`).

---

## Data Directory

```
my-memory/
  engram.yaml            # Configuration
  engram.db              # SQLite (messages, traces, events, sessions + FTS5)
  soul/
    SOUL.md              # Identity document
    journal/             # Dated journal entries
  semantic/
    identities.yaml      # Alias resolution
    preferences.yaml     # Likes, dislikes, uncertainties
    boundaries.yaml      # Behavioral boundaries
    trust.yaml           # Trust tiers
    contradictions.yaml  # Held contradictions
    relationships/       # Per-person relationship files
  procedural/            # Skill files
  safety/
    influence_log.yaml   # Manipulation attempt log
    injuries.yaml        # Psychological injury tracker
  personality/           # Big Five profile + history
  emotional/             # VAD state + events
  introspection/         # Meta-consciousness snapshots
  consciousness/         # Identity episodes + belief evolution
  embeddings/            # ChromaDB vector store
```

All semantic files are human-readable Markdown and YAML.

---

## Development

```bash
git clone https://github.com/Relic-Studios/engram.git
cd engram
pip install -e ".[all,dev]"
pytest                    # 606 tests
pytest -v --tb=short      # Verbose output
```

### Test Coverage

- **606 tests** across 20 test files
- Trust system: 74 tests
- Server tools: 83 tests (all 40 tools)
- Consolidation: 97 tests
- New subsystems: 38 integration tests
- Personality: 15, Emotional: 12, Workspace: 18, Introspection: 13, Consciousness: 19, Runtime: 14
- Plus: pipeline, episodic, semantic, procedural, signal, safety, system, journal, grounding, types

---

## License

MIT License. See [LICENSE](LICENSE).

## Author

**[Aidan Poole](https://github.com/AidanPoole)** / [Relic Studios](https://github.com/Relic-Studios)
