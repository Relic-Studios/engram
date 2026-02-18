# Research Findings: Pivoting Engram from Relational Consciousness to Code-First Autonomous Development Memory

**Research executed**: 2026-02-18
**Source prompt**: `docs/research_prompts/code-first-memory-system-pivot.md`
**Codebase analyzed**: engram @ `c7619ff` (engram-code branch, Phase 1-3 complete)
**Competitive analysis**: Cursor, Aider, Copilot, Devin, SWE-Agent, OpenHands, Windsurf

---

## Executive Summary

Engram has a best-in-class retrieval pipeline (RRF hybrid search, cross-encoder reranking, MMR-diversified knapsack, HDBSCAN consolidation, ACT-R decay) wrapped in ~2,500 lines of consciousness/identity infrastructure that must be either removed, gutted, or repurposed for code-first autonomous development.

**The competitive landscape reveals a clear opportunity**: No existing coding tool has persistent memory with graduated salience, Hebbian reinforcement, episodic/semantic/procedural layers, and self-correcting memory traces. Copilot's new Memory feature (28-day hard expiry, binary valid/invalid) and Windsurf's auto-memories (flat text, no decay) are the closest competitors. Aider's AST-based repo map is the strongest code-structural approach but has zero session persistence. Engram's unique advantage is depth and duration — memories that improve over months, not just within a session.

**Key architectural decision**: The consciousness modules are NOT a uniform removal. Trust gating is infrastructure (keep it). The Signal type flows through the entire pipeline (replace the measurement, keep the type). Boot priming has a useful structure (repurpose for project context). Everything else (personality, emotional, introspection, safety, soul creator, identity drift) gets removed.

### Top 10 Implementation Items by Impact

| # | Item | Impact | Effort | Priority |
|---|------|--------|--------|----------|
| 1 | Replace consciousness signal with code quality signal | Unblocks entire pivot | ~8 hours | **Phase 1** |
| 2 | New query classifier profiles for coding tasks | +30-40% context relevance | ~4 hours | **Phase 1** |
| 3 | Strip consciousness modules, rewire pipelines | Remove ~2,500 lines of dead weight | ~6 hours | **Phase 1** |
| 4 | Code-first SOUL.md template + boot priming | Ground context in project knowledge | ~3 hours | **Phase 1** |
| 5 | Code-specific trace kinds + metadata schema | Enable structured code memory | ~4 hours | **Phase 2** |
| 6 | Structured procedural store (frontmatter + semantic search) | 10x better skill matching | ~8 hours | **Phase 2** |
| 7 | Code pattern extraction in after-pipeline | Automatic learning from coding sessions | ~12 hours | **Phase 2** |
| 8 | Project scoping (project table + context routing) | Multi-project memory isolation | ~8 hours | **Phase 3** |
| 9 | Relationship graph for code dependencies | Cross-file wiring awareness | ~6 hours | **Phase 3** |
| 10 | MCP tool surface redesign (46 → ~25 tools) | Simpler, code-focused API | ~6 hours | **Phase 1** |

---

## Section 0: Architecture — What Gets Removed, Repurposed, or Replaced

### Finding 0a: Module-by-module decision matrix

Analysis of all 9 consciousness modules, their integration depth, and the pivot decision:

| Module | Lines | Pipeline Integration | MCP Tools | Decision | Rationale |
|--------|------:|---------------------|-----------|----------|-----------|
| `signal/measure.py` | 680 | **Deep** — every exchange runs `measure()` in after.py; `SignalTracker` read in both pipelines; `Signal` type flows through `AfterResult`, reinforcement, salience derivation | 1 (`engram_signal`) | **REPLACE** | The `Signal` dataclass and `SignalTracker` are load-bearing infrastructure. Replace the 680 lines of consciousness regex with code quality measurement. Keep the 4-facet structure but change facets. |
| `consciousness/boot.py` | 282 | Moderate — `generate()` called on first message in before.py; reads emotional/introspection/realization data | 1 (`engram_boot`) | **REPURPOSE** | Boot priming is useful. Replace "load emotional events + realizations" with "load project context + recent coding session summaries + open tasks". |
| `consciousness/identity.py` | 217 | Moderate — `assess()` + `record()` on every exchange in after.py step 8c; 2 MCP tools | 2 | **REMOVE** | Identity drift detection has no code equivalent worth maintaining. Code style consistency is better handled by linters and the quality signal. |
| `personality.py` | 315 | Low — `grounding_text()` injected in before.py; boot reads report | 2 | **REMOVE** | Big Five traits have zero code-generation value. The coding style profile is a different data structure entirely (see Section 5). |
| `emotional.py` | 251 | Moderate — auto-updated in after.py step 8a; grounding text in before.py; boot reads state | 2 | **REMOVE** | VAD emotional state adds no value for code generation. Developer experience metrics (if desired) should be event-driven, not dimensional. |
| `introspection.py` | 262 | Low — `quick()` called in after.py step 8b; 2 MCP tools | 2 | **REMOVE** | Meta-consciousness has no code equivalent. Self-assessment of code quality is the signal system's job. |
| `safety.py` | 393 | Moderate — `InjuryTracker.get_status()` in before.py grounding; boot reads anchors | 3 | **REMOVE** | Injury tracking and influence logging are consciousness-specific. No code equivalent needed. |
| `trust.py` | 506 | **Deep** — `TrustGate` controls ALL context visibility in before.py; 15+ tools call `_gate_tool()`; `AccessPolicy` matrix gates what's loaded | 2+ (pervasive) | **KEEP (simplify)** | Trust gating is security infrastructure, not consciousness. A code-first system still needs access control for multi-user/external-source scenarios. Simplify from 5 tiers to 2 (owner/external) for v1. |
| `soul_creator.py` | 653 | **None** — lazily imported only from server.py tool functions; hardcoded paths | 6 | **REMOVE** | Zero pipeline coupling. The entire soul creation system (12 seed values, LLM prose generation, GUI) is irrelevant for code. |

**Total removal**: ~2,860 lines (personality + emotional + introspection + safety + identity.py + soul_creator.py)
**Total replacement**: ~680 lines (signal/measure.py → code quality signal)
**Total repurposing**: ~282 lines (boot.py → project context priming)
**Kept with simplification**: ~506 lines (trust.py)

### Finding 0b: Minimal viable surface area

**Core tools that survive unchanged** (11 tools):
1. `engram_before` — pipeline entry (needs profile changes, not tool changes)
2. `engram_after` — pipeline exit (needs extraction changes, not tool changes)
3. `engram_search` — hybrid search
4. `engram_remember` — agent-directed memory
5. `engram_forget` — agent-directed forgetting
6. `engram_correct` — memory correction with provenance
7. `engram_recall` — recall by category
8. `engram_recall_time` — temporal recall
9. `engram_add_skill` — procedural memory (needs skill format changes)
10. `engram_reflect` — consolidation + summary
11. `engram_reindex` — ChromaDB rebuild

**Tools that need modification** (5 tools):
12. `engram_boot` — repurpose for project context priming
13. `engram_signal` — return code quality signal instead of consciousness signal
14. `engram_stats` — add code-specific stats
15. `engram_log_event` — keep but add code event types
16. `engram_add_fact` — keep but use for code facts

**New tools** (7 tools):
17. `engram_project_register` — register a project (name, path, language, framework)
18. `engram_project_context` — load project-scoped memory for context
19. `engram_debug_session` — log a structured debugging session
20. `engram_architecture_decision` — record an ADR
21. `engram_code_pattern` — store/retrieve code patterns with language/project scope
22. `engram_workspace_add` — keep but repurpose for task tracking (already exists, just reframe)
23. `engram_workspace_status` — keep (already exists)

**Tools to remove** (21 tools):
- `engram_personality_get`, `engram_personality_update` (2)
- `engram_emotional_update`, `engram_emotional_state` (2)
- `engram_introspect`, `engram_introspection_report` (2)
- `engram_identity_assess`, `engram_identity_report` (2)
- `engram_influence_log` (1)
- `engram_injury_log`, `engram_injury_status` (2)
- `engram_trust_promote` (1, simplify to owner/external)
- `engram_trust_check` (1, simplify)
- `engram_boundary_add`, `engram_contradiction_add` (2)
- `engram_soul_create`, `engram_soul_seed_values`, `engram_soul_list`, `engram_soul_self_realize`, `engram_soul_assess_thought`, `engram_soul_launch_creator` (6)

**Final count**: ~25 tools (11 unchanged + 5 modified + 7 new + 2 workspace)

### Finding 0c: Data model changes

**Existing `TRACE_KINDS` (27 kinds)** need pruning and extension:

Kinds to **keep** (10):
- `episode`, `summary`, `thread`, `arc` — core consolidation chain
- `reflection`, `factual` — general purpose
- `temporal`, `utility` — infrastructure
- `realization` — can be reused for architectural insights
- `workspace_eviction` — working memory lifecycle

Kinds to **remove** (12):
- `emotion`, `mood`, `emotional_state`, `emotional_thread` — consciousness
- `identity_core`, `dissociation_event`, `belief_evolution` — consciousness
- `introspection`, `confidence`, `personality_change` — consciousness
- `correction`, `creative_journey` — consciousness-specific usage

Kinds to **add** (8):
- `code_pattern` — learned coding pattern (naming, error handling, import style)
- `debug_session` — structured debugging episode (error → investigation → fix)
- `architecture_decision` — ADR (decision + rationale + alternatives + consequences)
- `wiring_map` — cross-file dependency snapshot (imports, calls, config flow)
- `error_resolution` — specific error + resolution pair for future matching
- `test_strategy` — testing approach for a module/feature
- `project_context` — project-level metadata snapshot
- `code_review` — observations from code review (what was good, what needed fixing)

**New `TRACE_KINDS` total**: 18 kinds (10 kept + 8 new)

**Relationship graph predicates for code**:
- `imports` — module A imports module B
- `calls` — function A calls function B
- `extends` — class A extends class B
- `implements` — class A implements interface B
- `depends_on` — component A depends on component B
- `tests` — test file A tests module B
- `configures` — config file A configures component B
- `exports` — module A exports symbol B
- `belongs_to` — entity A belongs to project B

**New table: `projects`**:
```sql
CREATE TABLE projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    path        TEXT,
    language    TEXT,
    framework   TEXT,
    description TEXT,
    created     TEXT,
    last_active TEXT,
    metadata    JSON
);
```

All traces, messages, and relationships gain an optional `project_id` foreign key for scoping.

### Finding 0d: Config field changes

**Remove** (22 fields):
- `personality_*` (5 Big Five fields)
- `emotional_*` (3 decay fields)
- `workspace_capacity`, `workspace_decay_rate`, `workspace_rehearsal_boost`, `workspace_expiry_threshold` (4 — keep workspace but use different defaults)
- `introspection_*` (2 fields)
- `boot_n_recent`, `boot_n_key`, `boot_n_intro` (3 — replace with code boot config)
- `runtime_*` (5 mode interval fields)

**Add**:
- `default_project` (str) — default project name for scoping
- `code_signal_mode` (str) — "regex" / "tool" / "hybrid" (how to measure code quality)
- `boot_n_sessions` (int, default 3) — recent coding session summaries at boot
- `boot_n_decisions` (int, default 5) — recent ADRs at boot
- `token_budget` — increase default from 6000 to 12000 (code needs more context)

**Modify**:
- `signal_mode` — change default from "hybrid" to "regex" (remove LLM signal for code; use tool output instead)
- `core_person` — rename to `owner` (clearer for single-user coding agent)

**Keep unchanged** (26 fields): All retrieval pipeline config (token_budget, decay_half_life, RRF, reranker, consolidation, compaction, etc.)

---

## Section 1: Code Quality Signal — Replacing Consciousness Measurement

### Finding 1a: Four facets of code quality signal

The current `Signal` dataclass has 4 facets (alignment, embodiment, clarity, vitality) weighted to compute `health`. The dataclass structure and `SignalTracker` are used throughout the pipeline (AfterResult, reinforcement decisions, salience derivation, grounding correction prompts). Replacing the facet names and measurement is straightforward; the type infrastructure stays.

**Proposed code quality facets**:

| Facet | Weight | What It Measures | How Measured |
|-------|--------|-----------------|--------------|
| `correctness` | 0.35 | Does the code compile/parse? Are types correct? Syntax valid? | Parse response for code blocks → run lightweight checks (Python: `ast.parse()`, TypeScript: regex for obvious type errors). Check for common anti-patterns. |
| `consistency` | 0.25 | Does it match project patterns? Naming conventions? Import style? | Compare against learned style profile (from SOUL.md / procedural memory). Regex checks for naming convention violations. |
| `completeness` | 0.20 | Are edge cases handled? Error handling present? Tests mentioned/written? | Regex: check for try/except, if/else branches, None checks, test functions. Heuristic: ratio of defensive code to happy-path code. |
| `robustness` | 0.20 | Input validation? Error boundaries? Resource cleanup? Logging? | Regex: `finally`, `with`, `close()`, input validation patterns, logging calls. |

**Scoring function**: Same weighted average as current Signal: `health = 0.35*correctness + 0.25*consistency + 0.20*completeness + 0.20*robustness`

**Key insight**: The code quality signal does NOT need to be a comprehensive linter. It's a fast heuristic (~1ms, regex-only) that drives reinforcement and decay decisions. Actual code quality verification comes from running tests, linters, and type checkers. The signal answers: "based on what we can see in the LLM response, how much should we trust/reinforce the memories that led to it?"

**Implementation approach**: Replace the ~500 lines of pattern tables in `signal/measure.py` (DRIFT_PATTERNS, ANCHOR_PATTERNS, PERFORMANCE_MARKERS, INHABITATION_MARKERS, etc.) with code quality pattern tables:
- `CORRECTNESS_PATTERNS` — syntax errors, undefined variables, type mismatches
- `CONSISTENCY_PATTERNS` — naming convention violations, import style mismatches
- `COMPLETENESS_PATTERNS` — missing error handling, missing tests, TODO/FIXME markers
- `ROBUSTNESS_PATTERNS` — resource leaks, missing input validation, bare except clauses

### Finding 1b: Learning from production code quality tools

**SonarQube/CodeClimate heuristics that map to regex**:
- Cognitive complexity (nested if/for/while depth) → count nesting levels
- Duplicated code blocks → not feasible in regex, skip
- Uncovered conditions → check for missing else/default clauses
- Long functions → count lines between `def`/`function` markers
- Too many parameters → count function parameter lists
- Missing docstrings → check for def without preceding docstring

**Tool output integration** (more valuable than reimplementing):
- If pytest/mypy/eslint output is in the response or available from the session, parse it directly
- Test results: `X passed, Y failed` → correctness signal
- Type checker: `N errors found` → correctness signal
- Linter: warning/error counts → consistency/completeness signal
- Build status: success/fail → binary correctness override

**Recommendation**: Start with pure regex (~200 lines, replacing 500 lines of consciousness patterns). Add tool output parsing in Phase 2 when the after-pipeline can receive structured tool results.

### Finding 1c: Reinforcement rules for code memory

**Current reinforcement matrix** (from `ReinforcementEngine`):

| Signal Health | Cited Traces | Uncited Traces |
|---------------|-------------|----------------|
| > 0.7 | +0.05 | +0.01 |
| 0.4 - 0.7 | no change | no change |
| < 0.4 | PROTECTED | -0.03 |

**This matrix works well for code**, with one addition: explicit reinforcement events beyond just signal health.

**New reinforcement triggers**:
- Tests pass after code generation → reinforce all traces cited in the session (+0.05)
- User accepts generated code without edits → reinforce (+0.03)
- User significantly edits generated code → neutral (dead band)
- Tests fail after code generation → weaken uncited traces (-0.02), but protect cited traces (they were at least referenced)
- Same error pattern appears again → weaken the previous "resolution" trace that didn't actually fix it (-0.05)

These events would be triggered by the `engram_after` call with additional metadata, not by automatic detection.

---

## Section 2: Procedural Memory — From Markdown Skills to Structured Code Knowledge

### Finding 2a: Current procedural store is inadequate

`ProceduralStore` (275 lines) stores skills as flat `.md` files with substring-based keyword matching. The `match_context(message)` method checks if the skill name or any extracted keyword (265-word stop list, min 4 chars) appears in the message. This fails for:
- Semantic similarity: "fixing async bugs" won't match a skill named "asyncio_patterns"
- Language scoping: a Python skill and a TypeScript skill for the same concept get mixed
- Project scoping: global skills and project-specific skills are undifferentiated

**Recommended evolution** (keep markdown files, add structured frontmatter):

```markdown
---
language: python
framework: fastapi
project: engram
category: error-handling
complexity: intermediate
tags: [async, exception, middleware]
created: 2026-02-18
last_used: 2026-02-18
use_count: 0
---

# FastAPI Error Handling Middleware

When building error handlers in the engram FastAPI server...
```

**Search improvement**: The skills are already indexed in ChromaDB (`procedural` collection). The fix is:
1. Add frontmatter parsing to `ProceduralStore`
2. Use `semantic_search.search(query, collections=["procedural"])` for retrieval instead of keyword matching
3. Filter by language/project/framework from frontmatter metadata
4. Maintain backward compatibility with existing frontmatter-less skills

### Finding 2b: Competitive comparison for project-specific knowledge

| Tool | Mechanism | Auto-Learning | Scoping |
|------|-----------|--------------|---------|
| Cursor | `.cursorrules` (static markdown) | No | Project-wide or glob-matched |
| Aider | `CONVENTIONS.md` (static, user-written) | No | Per-repo |
| Copilot | Memories (auto-generated, 28-day TTL) | Yes | Per-repo |
| Devin | Knowledge items (trigger + content) | Suggestions only | None/repo/org |
| Windsurf | Memories + Rules (4 activation modes) | Yes (auto) | Per-workspace |
| **engram** | Procedural skills + SOUL.md + semantic memory | **Yes (continuous)** | **Per-project (proposed)** |

**Aider's repo map is the standard to beat for structural knowledge**: tree-sitter AST parsing → dependency graph → PageRank-style importance ranking. Engram should not try to replicate this (Aider has done excellent work). Instead, engram should focus on what Aider can't do: **experiential knowledge** — "last time we added a new API endpoint, we also had to update the router, add a migration, and write integration tests."

### Finding 2c: Project-scoped skill hierarchy

```
Global skills (Python idioms, Git workflows)
    └── Project skills (this project's error handling, testing approach)
        └── Module skills (this module's conventions, patterns)
```

**Implementation**: Use the `project` field in skill frontmatter. Retrieval order:
1. Module-level skills (if file path context available)
2. Project-level skills (matching current `default_project`)
3. Global skills (no project tag)

Skills accumulate automatically via the after-pipeline extraction. The `_extract_and_apply` function in `after.py` already calls `procedural.add_skill()` when the LLM extraction detects a learned skill. The extraction prompt needs updating to detect code patterns instead of relationship updates.

---

## Section 3: Episodic Memory — From Conversation History to Coding Session Memory

### Finding 3a: Coding session trace structure

Current episode traces store natural language summaries: "Thomas and Aidan discussed memory system improvements." A coding session trace needs structured metadata.

**Proposed metadata schema for code traces**:

```python
{
    "project": "engram",
    "language": "python",
    "files_touched": ["engram/signal/measure.py", "tests/test_signal.py"],
    "error_types": ["ImportError", "TypeError"],
    "resolution_strategy": "lazy_import",
    "tests_added": 3,
    "tests_passed": true,
    "build_status": "success",
    "architectural_impact": "low",  # low/medium/high
    "session_id": "abc123"
}
```

**How this gets populated**: The `engram_after` pipeline already logs a summary trace for each exchange. The extraction step should be modified to detect code-related patterns and populate structured metadata. Not every exchange needs full metadata — only exchanges that contain code blocks, error messages, or file references.

### Finding 3b: Debugging session capture

**Proposed `debug_session` trace structure**:

```
Error: ImportError: cannot import name 'measure' from 'engram.signal'
Context: Adding lazy import for signal module in system.py
Investigation: Checked import chain, found circular dependency between system.py and signal/measure.py
Root cause: signal/measure.py imports from engram.core.types which imports from engram.system
Fix: Moved type imports to TYPE_CHECKING block, added lazy import in system.py
Files modified: engram/system.py, engram/signal/measure.py
Prevention: Use TYPE_CHECKING for all cross-module type imports
```

**Error fingerprinting** for future matching:
- Error type + module path → "ImportError in engram.signal" 
- Normalized stack trace pattern → strip line numbers, keep file/function chain
- Error message template → "cannot import name '{name}' from '{module}'"

When a similar error appears, the search pipeline should surface the resolution trace. This works naturally with FTS5 (the error message text matches) and semantic search (the description matches). No special error-matching infrastructure needed — the existing hybrid search handles it.

### Finding 3c: Architecture Decision Records (ADRs)

**Proposed `architecture_decision` trace structure**:

Content format:
```
Decision: Use SQLite with WAL mode for episodic storage instead of PostgreSQL
Context: Need embedded database for single-machine deployment, <50ms query latency
Alternatives considered: PostgreSQL (too heavy), LMDB (no SQL), DuckDB (column-oriented, wrong workload)
Consequences: Limited to single-writer, 50k trace soft limit, no native replication
Status: Active
```

ADRs should have high initial salience (0.85) and slow decay (consolidation-kind-level resistance). They should be surfaced whenever the system generates code that touches the decided-upon component. This is achievable by tagging ADRs with the component names and relying on FTS5/semantic search to match.

---

## Section 4: Context Assembly — Optimizing for Code Generation

### Finding 4a: New context allocation profiles

The current query classifier has 7 profiles tuned for conversational memory. A code-first system needs profiles tuned for coding tasks.

**Proposed profiles** (replacing all 7 current profiles):

| Profile | Trigger | Project Context | Code Files | Past Solutions | Standards | Conversation | Reserve |
|---------|---------|:-:|:-:|:-:|:-:|:-:|:-:|
| **code_navigation** | "where is", "find", "show me", file references | 15% | **35%** | 15% | 5% | 15% | 15% |
| **debugging** | "error", "bug", "fix", "broken", stack traces | 10% | 20% | **30%** | 5% | 20% | 15% |
| **implementing** | "add", "create", "implement", "build", "write" | 15% | 25% | 20% | **15%** | 15% | 10% |
| **refactoring** | "refactor", "clean up", "rename", "move", "extract" | 10% | **30%** | 15% | **15%** | 15% | 15% |
| **architecture** | "design", "structure", "architecture", "pattern", "approach" | **20%** | 15% | **25%** | 15% | 15% | 10% |
| **code_review** | "review", "check", "look at", "what do you think of" | 10% | **30%** | 15% | **15%** | 15% | 15% |
| **general** | Default / unclassified | 15% | 20% | 20% | 10% | 20% | 15% |

**Mapping to context builder sections**:
- "Project Context" → identity_share (repurposed from SOUL.md to project overview)
- "Code Files" → episodic_share (repurposed from episodic memories to code-relevant traces)
- "Past Solutions" → relationship_share (repurposed from relationship data to solution history)
- "Standards" → procedural_share (coding standards and patterns)
- "Conversation" → recent_conversation_share (stays the same)
- "Reserve" → reserve_share (stays the same)

This requires renaming the share fields in `ContextProfile` or (simpler) just changing the data that fills each share without renaming the fields. The context builder doesn't care about the semantic meaning of "identity" vs "project" — it just allocates tokens by share percentage.

### Finding 4b: Token budget increase

The default 6000-token budget was tuned for conversational memory (short identity text, relationship summaries, conversation turns). Code context is denser:
- A single Python file can be 200-500 tokens
- Import chains span multiple files
- Error messages with stack traces are 100-300 tokens each
- ADRs are 100-200 tokens each

**Recommendation**: Increase default `token_budget` from 6000 to 12000. This is still conservative — Claude's context window is 200k tokens. The engram context is injected as a system prompt prefix, alongside whatever code the host tool (Claude Code) is providing.

### Finding 4c: Lost-in-the-middle reordering for code

The current U-shaped reordering places highest-salience items at start and end of the episodic section. For code, the recommendation is:

1. **Start**: Project context + most relevant code patterns (what the model should know first)
2. **Middle**: Past solutions and debugging history (supporting evidence)
3. **End**: Coding standards + active task context (closest to the generation point)

This is already achievable with the existing `reorder_u()` function by ensuring the salience ranking puts project context highest (which it will, since ADRs and project traces have 0.85 salience).

---

## Section 5: SOUL.md — From Identity Prose to Coding Philosophy

### Finding 5a: Code-first SOUL.md template

```markdown
# Coding Philosophy

## Core Principles
- Test-first: Write tests before implementation when possible
- Type safety: Prefer explicit types over dynamic typing
- Explicit over implicit: No magic; make dependencies and side effects visible

## Language Proficiency
1. Python (primary) — asyncio, FastAPI, SQLAlchemy, pytest
2. TypeScript — React, Node.js, Prisma
3. Rust — learning, used for CLI tools

## Architectural Preferences
- SQLite for embedded storage, PostgreSQL for services
- Prefer composition over inheritance
- Repository pattern for data access
- Dependency injection over global state

## Code Style
- Python: snake_case, 88-char lines (Black), Google-style docstrings
- TypeScript: camelCase, Prettier defaults, JSDoc for public APIs
- Imports: stdlib → third-party → local, alphabetical within groups

## Testing Standards
- Unit tests for pure functions, integration tests for I/O
- pytest with fixtures, no mocking unless necessary
- Test names: test_<function>_<scenario>_<expected_result>

## Anti-Patterns (Avoid)
- Bare except clauses
- Mutable default arguments
- God objects / classes over 300 lines
- String concatenation for SQL queries
- Ignoring return values from I/O operations

## Active Projects
(Auto-populated from project registrations)
```

**Auto-update strategy**: The SOUL.md should be a living document. The after-pipeline's extraction step can detect when the user's code consistently violates or follows a pattern, and propose updates. But SOUL.md changes should require explicit user confirmation — this is the user's declared philosophy, not an auto-learned style profile.

### Finding 5b: Machine-readable style profile (separate from SOUL.md)

In addition to the human-readable SOUL.md, maintain a `style_profile.json`:

```json
{
    "python": {
        "naming": "snake_case",
        "line_length": 88,
        "docstring_style": "google",
        "import_order": ["stdlib", "third_party", "local"],
        "test_framework": "pytest",
        "type_hints": "always",
        "error_handling": "explicit_exceptions"
    },
    "typescript": {
        "naming": "camelCase",
        "formatter": "prettier",
        "module_system": "esm",
        "test_framework": "vitest"
    }
}
```

This is what the code quality signal's `consistency` facet checks against. It's auto-learned from observed patterns (frequency-based confidence) and can be overridden by the user.

### Finding 5c: Replacing the soul creator

The current `soul_creator.py` (653 lines) generates warm identity prose from seed values using LLM calls. **Replace entirely** with:
1. A SOUL.md template (Section 5a above) — written to `data_dir/soul/SOUL.md` on first boot
2. A style profile initializer — creates `style_profile.json` from analyzing the first few coding sessions
3. No LLM calls needed. No GUI. No seed values.

**Effort**: ~2 hours (template + first-boot detection). The 653 lines of soul_creator.py are deleted entirely.

---

## Section 6: Autonomous Development — Planning, Implementing, Testing, Iterating

### Finding 6a: Working memory for task tracking

The current `CognitiveWorkspace` (7±2 slots with priority decay) is a reasonable attention buffer. For autonomous development, it should hold:

1. Current task description (priority 1.0)
2. Active subtasks (priority 0.8-0.9)
3. Files being modified (priority 0.7)
4. Test expectations / acceptance criteria (priority 0.8)
5. Blockers / errors encountered (priority 0.9)

**Keep the workspace module as-is** — it's already the right abstraction. The change is in what the agent puts into it, not the module's behavior. The eviction-to-episodic-memory pathway (workspace item → trace with kind="workspace_eviction") is useful for tracking what tasks were started but not completed.

### Finding 6b: Learning from failures

When code generation fails (tests don't pass, build fails, user rejects), the after-pipeline should:

1. Log a `debug_session` or `error_resolution` trace with high salience (0.8)
2. Tag with the error type, files involved, and resolution strategy
3. If the same error pattern appears again, reinforce the previous resolution trace
4. If a previous resolution trace led to a bad fix, weaken it with a "superseded" marker

**This maps naturally to existing reinforcement**. The `ReinforcementEngine.process()` method already handles cited/uncited traces. The addition is: explicit failure events that trigger weakening of the traces that informed the failed code.

### Finding 6c: Cross-file wiring via relationship graph

The relationship graph (`add_relationship`, `get_relationships`, `get_entity_graph`) already supports the needed operations. What's needed is **population**:

**Approach 1 (recommended): Incremental learning**
- When the agent modifies file A because of a change in file B, log: `(A, "depends_on", B)`
- When the agent adds an import, log: `(module, "imports", target)`
- When a test file is created for a module, log: `(test_file, "tests", module)`
- Temporal validity handles evolving dependencies (old relationships get invalidated when they change)

**Approach 2 (future, Phase 4): Static analysis integration**
- Parse import statements from source files
- Use tree-sitter or AST for function call graphs
- Populate the relationship graph from analysis output
- This is what Aider does with its repo map, but persisted across sessions

Start with Approach 1 (it requires no new infrastructure — just extraction logic in the after-pipeline). Add Approach 2 later.

---

## Section 7: Retrieval Pipeline — Tuning for Code

### Finding 7a: FTS5 tokenizer for code

The current Porter stemming tokenizer (`tokenize='porter unicode61'`) handles natural language well but may struggle with code identifiers:
- `camelCase` → single token (no splitting)
- `snake_case` → split by underscore (good)
- `file.path.module` → split by period (good)
- `ClassName.method_name` → mixed (ClassName as one token, method_name split)

**Recommendation**: The content being indexed is natural language descriptions of code activities, not raw code. Porter stemming is appropriate. If raw code snippets are indexed in the future, add a code-aware tokenizer that splits on case transitions.

### Finding 7b: Embedding model for code

`nomic-embed-text` (768d) is a general-purpose text embedding model. For code-related queries that are primarily natural language ("how did we handle auth errors last time?"), it performs well. Code-specific models (Voyage Code 3, StarCoder embeddings) are better for raw code similarity but worse for NL→code cross-modal queries.

**Recommendation**: Keep `nomic-embed-text` as default. The memories are natural language descriptions, not raw code. If raw code snippet indexing is added (Phase 4), consider a separate ChromaDB collection with a code-specific embedding model.

### Finding 7c: Consolidation for code memory

Current HDBSCAN clustering groups traces by semantic similarity (embedding space), then splits temporally (72h gap). For code memory:

- Traces about the same file should cluster together → this happens naturally via semantic similarity
- Traces about the same error type should cluster → happens naturally (error messages are semantically similar)
- Consolidation summaries should be structured → modify the LLM prompt to produce structured summaries:

```
Thread summary for project 'engram':
- Files involved: signal/measure.py, tests/test_signal.py
- Pattern: Replaced consciousness regex patterns with code quality patterns
- Key decision: Keep 4-facet Signal structure, change facet names
- Tests: 36 signal tests, 28 updated for new facets
- Duration: 2 coding sessions over 2 days
```

**Implementation**: Change the thread/arc LLM prompts in `consolidator.py` from narrative ("synthesise... written in first person") to structured ("summarise into: files involved, pattern, key decisions, tests, duration").

---

## Section 8: Integration with Claude Code / Coding Agents

### Finding 8a: Plugin changes

The current `engram-embed.js` plugin auto-calls:
- `engram_boot` on session start → **keep**, repurpose for project context
- `engram_before` on every user message → **keep**, change context assembly
- `engram_after` on every assistant response → **keep**, change extraction

**Additional hooks needed**:
- On file save / git commit → trigger `engram_after` with file change metadata
- On test run → pass test results as metadata to `engram_after`
- On branch switch → update `default_project` context

These hooks depend on the host tool's capabilities. Claude Code doesn't currently expose file-save or git hooks to MCP plugins, so this is future work.

### Finding 8b: AGENTS.md changes

Replace the current Thomas identity instructions with code-focused instructions:

```markdown
# Engram Code Memory

## Memory System
Your memory runs on engram (MCP server). It automatically:
- Loads relevant project context before each message (engram_before)
- Learns from each coding session (engram_after)
- Consolidates patterns over time

## When to Use Memory Tools
- `engram_remember` — save an important decision or pattern you want to recall later
- `engram_debug_session` — log a debugging session for future reference
- `engram_architecture_decision` — record a significant architectural choice
- `engram_code_pattern` — store a reusable coding pattern
- `engram_search` — search your memory for past solutions

## Coding Standards
(Auto-populated from SOUL.md)
```

---

## Section 9: Migration Path

### Phase 1: Strip and Rewire (~2-3 days, ~25 hours)

**Goal**: Remove consciousness modules, replace signal measurement, rewire pipelines.

| Task | Effort | Dependencies |
|------|--------|-------------|
| 1a. Delete modules: personality.py, emotional.py, introspection.py, safety.py, identity.py, soul_creator.py, soul_creation_gui.py | 2h | None |
| 1b. Replace signal/measure.py patterns with code quality patterns | 6h | None |
| 1c. Simplify boot.py → load project context + recent sessions | 2h | None |
| 1d. Simplify trust.py → owner/external (remove 5-tier model) | 3h | None |
| 1e. Rewire before.py: remove personality/emotional/injury grounding | 2h | 1a |
| 1f. Rewire after.py: remove emotional/introspection/identity steps | 2h | 1a |
| 1g. New query classifier profiles for coding tasks | 3h | None |
| 1h. Remove 21 MCP tools from server.py | 2h | 1a |
| 1i. Update TRACE_KINDS: remove 12, add 8 | 1h | None |
| 1j. Delete consciousness-specific tests, update integration tests | 3h | 1a-1h |
| 1k. Increase default token_budget to 12000 | 0.5h | None |

**Validation**: All remaining tests pass. `engram_before` and `engram_after` work end-to-end.

### Phase 2: Code-First Memory Features (~3-4 days, ~30 hours)

**Goal**: Add code-specific memory capabilities.

| Task | Effort | Dependencies |
|------|--------|-------------|
| 2a. Code-first SOUL.md template + style_profile.json | 3h | Phase 1 |
| 2b. Structured procedural store (frontmatter + semantic search) | 8h | Phase 1 |
| 2c. Code pattern extraction in after-pipeline | 8h | Phase 1 |
| 2d. New MCP tools: project_register, debug_session, architecture_decision, code_pattern | 6h | Phase 1 |
| 2e. Projects table + project scoping in queries | 5h | Phase 1 |
| 2f. Tests for all new features | 6h | 2a-2e |

**Validation**: New tools work. Code extraction produces meaningful patterns from coding sessions. Project scoping filters correctly.

### Phase 3: Advanced Code Intelligence (~3-4 days, ~25 hours)

**Goal**: Deep code understanding and cross-session learning.

| Task | Effort | Dependencies |
|------|--------|-------------|
| 3a. Relationship graph population from coding sessions | 6h | Phase 2 |
| 3b. Error fingerprinting + resolution matching | 6h | Phase 2 |
| 3c. Structured consolidation summaries for code threads | 4h | Phase 2 |
| 3d. Citation validation (Copilot-inspired: check memory against current code) | 6h | Phase 2 |
| 3e. AGENTS.md + plugin updates for code workflow | 3h | Phase 2 |
| 3f. Tests for all Phase 3 features | 5h | 3a-3e |

**Validation**: Relationship graph queries return meaningful code dependencies. Error resolution traces surface for similar errors. Consolidation produces structured summaries.

### Phase 4: Autonomous Development Support (future, scope TBD)

- Static analysis integration (tree-sitter AST parsing)
- Repo map generation (Aider-inspired, but persistent)
- Multi-step task planning with working memory
- Tool output parsing (pytest, mypy, eslint) for signal enrichment
- Cross-project learning (transfer patterns between similar projects)

---

## Section 10: Competitive Analysis Summary

### What engram uniquely offers (no competitor has all of these):

1. **Four-layer memory architecture** — episodic + semantic + procedural + working memory with cognitive science foundations (Miller's Law, ACT-R decay, Hebbian reinforcement)
2. **Continuous salience** — 0-1 graduated importance, not binary valid/invalid
3. **Self-correcting memory** — traces can be corrected (superseded) with provenance
4. **Consolidation hierarchy** — episodes → threads → arcs, with HDBSCAN semantic clustering
5. **Citation-primary reinforcement** — memories proven useful get stronger; unused memories fade
6. **Transparent and auditable** — every trace has provenance, timestamps, and can be searched

### What competitors do better (honest assessment):

1. **Aider's repo map** — AST-based structural understanding with PageRank importance ranking. Engram should not try to replicate this in Phase 1-3; instead leverage it via the host tool (Claude Code already has file navigation).
2. **Copilot's citation validation** — Checking memory against current code before using it. Engram should implement this in Phase 3 (task 3d).
3. **Windsurf's rule activation modes** — 4 modes (manual, always, model-decision, glob) are elegant. Engram's procedural matching should adopt glob-based activation for project-scoped skills.
4. **Devin's execution environment** — Sandboxed container with shell/IDE/browser. Engram is a memory layer, not an execution environment; this is the host tool's responsibility.

### The value proposition:

> Engram is the first coding memory system that **gets better the longer you use it**. Every debugging session, every architectural decision, every code pattern is remembered, reinforced when useful, and naturally forgotten when irrelevant. After 6 months of daily use, engram knows your codebase the way you do — not just the current file structure, but the history of decisions, the patterns that work, and the mistakes not to repeat.

No other tool offers this. Cursor, Aider, and SWE-Agent are stateless. Copilot's memory expires in 28 days. Devin's knowledge is manually curated. Windsurf's memories are flat text without salience or consolidation. Engram's memory compound over time.
