# nano-hermes

Self-evolving memory, skill lifecycle, and Reflexion-based self-improvement extensions for [HKUDS/nanobot](https://github.com/HKUDS/nanobot).

Adds ten agent-facing tools plus a lifecycle hook that archives turns into a searchable SQLite index, runs Reflexion-style self-critique, maintains a Voyager-style embedding index over nanobot's skill library, supports a two-phase skill promotion system (draft → active → deprecated), and automatically rewrites chronically failing skills using a SkillForge-inspired pipeline.

Designed for low-resource hosts (Raspberry Pi) — no local embedding model, one SQLite file per workspace, hosted-embedding failover across DeepInfra, Together, and OpenRouter.

---

## Requirements

- **Python 3.11+** (tested on 3.14).
- **nanobot-ai** — the host framework.
- **One or more embedding API keys**: `DEEPINFRA_API_KEY`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`. Failover walks the chain in that order; any one is enough to start.

---

## Install

### On a Pi with `uv tool`

`uv tool` installs each CLI into its own isolated venv. nano-hermes must be the **primary** package so its `nano-hermes` entry point lands on `PATH`. nanobot-ai is pulled in as a `--with` extra.

```bash
# Install from a local clone (editable — changes take effect immediately):
uv tool install --editable /path/to/nano-hermes --with nanobot-ai

# Or from PyPI:
uv tool install nano-hermes --with nanobot-ai
```

If you previously had `nanobot-ai` installed as a standalone tool you can remove it to avoid accidentally using the plugin-free `nanobot` command:
```bash
uv tool uninstall nanobot-ai
```

After install, `nano-hermes` is on `PATH`:

**Use `nano-hermes` in place of `nanobot`** for any subcommand:
```bash
nano-hermes agent         # interactive REPL
nano-hermes gateway       # multi-channel gateway
nano-hermes status
nano-hermes --help        # full nanobot help, unchanged
```

### Raspberry Pi 3B+ notes

The 1 GB / ARM Cortex-A53 / microSD profile is the bottom-end target. It works, but watch for:

- **Use the 64-bit Raspberry Pi OS.** `sqlite-vec` and `tiktoken` (pulled in by nanobot) have spotty ARM32 wheels; on 32-bit you fall back to a from-source build that can OOM on 1 GB.
- **First install needs a build toolchain.** `readability-lxml` (nanobot dep) wants `libxml2-dev libxslt1-dev` plus `gcc`; install with `apt install build-essential libxml2-dev libxslt1-dev` before `uv tool install`.
- **Prefer a UHS-II / U3 microSD card.** SQLite WAL plus FTS5 and vec0 do a lot of small writes; a slow card is the most common source of turn-to-turn jitter.
- **RAM budget is tight.** Baseline (nanobot + nano-hermes hook + DB connection) is ~250-300 MB. Long `/goal` sessions with many tool calls can push past 500 MB — leave headroom for the OS. The Pi 4/5 is the better target for 24/7 use.
- **Keep `distill_max_chunks ≤ 150`.** The hub-cluster pass is O(N²); the default cap is tuned for this hardware.
- **Maintenance.** The retention purge runs on every session start and now also `VACUUM`s when rows were actually removed, so FTS5/vec0 fragmentation stays bounded without manual intervention.

### For developing nano-hermes itself

```bash
cd /path/to/nano-hermes
python3 -m venv .venv && source .venv/bin/activate
pip install -e /path/to/nanobot      # local nanobot checkout
pip install -e '.[dev]'              # nano-hermes + pytest + sqlite-vec
pytest -v                            # 636 tests, expect all green
```

If `sqlite-vec` has no wheel for your Python version (e.g. 3.14), build from source:
```bash
pip install sqlite-vec --no-binary :all:
```
or drop to Python 3.13 in the venv.

---

## Run

### Primary — the `nano-hermes` CLI

```bash
export DEEPINFRA_API_KEY=...
# Any one of the three is enough; more = failover.
export TOGETHER_API_KEY=...
export OPENROUTER_API_KEY=...

nano-hermes agent
```

The wrapper monkey-patches `AgentLoop.__init__` at import time and calls `nano_hermes.install(loop)` on every instance nanobot constructs. Your existing `~/.nanobot/config.json` is respected.

### Alternative — Python SDK (embedded use)

```python
import asyncio
from nanobot.nanobot import Nanobot
import nano_hermes

async def main() -> None:
    bot = Nanobot.from_config()
    nano_hermes.install(bot._loop)          # attach hook + register tools
    result = await bot.run("what's in my memory about trip planning?")
    print(result.content)

asyncio.run(main())
```

`install(loop)` returns the `NanoHermesHook` so you can inspect or detach it.

---

## Configuration

All defaults are locked in `NanoHermesConfig` — you only override what you want.

### Via environment variables

```bash
DEEPINFRA_API_KEY=sk-...      # embedding chain primary
TOGETHER_API_KEY=...          # failover
OPENROUTER_API_KEY=...        # failover
```

The state database lives under `<workspace>/nano_hermes/state.db`.

### Via config files (recommended for CLI use)

When `nano-hermes gateway` (or `nano-hermes agent`) starts, it calls `install()` with no config argument. nano-hermes automatically discovers and merges two JSON files:

| File | Scope |
|---|---|
| `~/.nanobot/nano_hermes.json` | user-level defaults (all workspaces) |
| `<workspace>/nano_hermes/config.json` | workspace-specific overrides |

Workspace values win on any key present in both files. Missing files are silently skipped. A fully-annotated example is at [`examples/nano_hermes.json`](examples/nano_hermes.json) in this repo.

**Quick start** — copy the example to your user config and edit as needed:

```bash
mkdir -p ~/.nanobot
cp /path/to/nano-hermes/examples/nano_hermes.json ~/.nanobot/nano_hermes.json
```

The most useful knob to set immediately is `rewrite_session_interval`:

```json
{
  "skill_stats": {
    "rewrite_session_interval": 5
  }
}
```

### Via install() config (Python SDK)

```python
nano_hermes.install(loop, config={
    # ── Memory budgets ──────────────────────────────────────────────────
    "memory": {
        "memory_md_tokens": 512,    # default 512  (~2000 English chars)
        "user_md_tokens": 320,      # default 320
        "soul_md_tokens": 384,      # default 384
        # memory_patch(action="consolidate") threshold — merge entries with
        # cosine similarity ≥ this. 0.92 collapses near-duplicates only.
        "consolidation_similarity_threshold": 0.92,
        # memory_patch(action="distill") — episodic→semantic hub detection.
        # A chunk cluster must span this many successful sessions to surface.
        "distill_hub_min_sessions": 2,
        "distill_max_chunks": 150,          # cap before O(N²) clustering
        "distill_cluster_threshold": 0.88,
    },

    # ── Retrieval ────────────────────────────────────────────────────────
    "retrieval": {
        "final_k": 8,               # default 8 — hits returned by session_search
        "mmr_lambda": 0.7,          # diversity reranking λ (1.0 = pure relevance)
    },

    # ── Reflexion nudges ─────────────────────────────────────────────────
    "reflection": {
        "threshold": 5.0,           # default 5.0 — lower = more nudges
        "recent_limit": 5,          # default 5 — max reflections injected per iter
        "global_inject_min_similarity": 0.60,  # cross-session injection threshold
    },
    "reflection_scope": "global",   # "session" (default) or "global"

    # ── Skill lifecycle & ranking ────────────────────────────────────────
    "skill_stats": {
        # Ranking mode for skill_search re-ranking.
        # "ucb1" (default) — exploration+exploitation bandit
        # "stat_weighted" — legacy success-rate boost
        # "off" — no stat adjustment
        "ranking_mode": "ucb1",
        "ucb1_coefficient": 0.05,           # UCB1 bandit coefficient

        "promotion_threshold": 3,           # successful uses → draft→active
        "deprecation_min_uses": 5,          # uses before deprecation check
        "deprecation_max_success_rate": 0.2,# below this rate → deprecated
        "max_skill_bytes": 262144,          # 256 KiB max per propose_skill call
        "diversity_similarity_threshold": 0.88,  # promotion diversity gate
        "skill_search_dedup_threshold": 0.82,    # search-time sibling collapsing

        # ── Failure-driven auto-rewriter (SkillForge) ───────────────────
        # Skills with failure_rate > threshold AND use_count >= min_uses are
        # automatically rewritten. Off until skills accumulate enough data.
        "rewrite_failure_threshold": 0.6,   # >60% failure rate triggers rewrite
        "rewrite_min_uses": 5,
        "rewrite_context_chunks": 5,        # failed-session chunks sent to LLM

        # ── GEPA iterative evolution ─────────────────────────────────────
        # Gentler first pass (lower threshold) before the rewriter.
        # Off by default — enable once you have ≥5 sessions of failure data.
        "gepa_enabled": False,
        "gepa_failure_threshold": 0.4,
        "gepa_min_uses": 5,
        "gepa_max_mutations": 3,
        "gepa_minibatch_size": 3,

        # ── Auto-evolution trigger ───────────────────────────────────────
        # Run GEPA (if enabled) then rewriter every N completed sessions.
        # 0 = disabled. Recommended starting value: 5–10.
        "rewrite_session_interval": 0,
    },

    # ── Trajectory replay ────────────────────────────────────────────────
    "trajectory": {
        "inject_context": False,            # inject similar past trajectory on iter 0
        "inject_min_similarity": 0.75,
    },
    "trajectory_retention_days": 45,

    # ── Workflow induction (Phase 4.1) ───────────────────────────────────
    # Off by default. Enable once you have ≥25 successful sessions.
    "workflow_induction": {
        "enabled": False,
        "min_cluster_size": 3,      # minimum trajectories to form a workflow candidate
        "max_trajectories": 100,
        "cluster_threshold": 0.85,
    },
})
```

---

## What the agent gets

Ten tools land on `loop.tools`:

| Tool | What it does |
|---|---|
| `memory_patch(action, slot?, ...)` | Edit `MEMORY.md` / `USER.md` / `SOUL.md`. `slot ∈ {memory, user, soul}`. `action ∈ {add, replace, remove, consolidate, distill}`. `consolidate`: embed entries, merge near-duplicates (cosine ≥ threshold), keeps longest entry per cluster — call when memory feels bloated. `distill`: find recurring themes across successful sessions and surface candidate facts for you to add to memory; `slot` is ignored. |
| `session_search(query, k=8)` | Hybrid FTS5 + embedding search over archived turn chunks. RRF fusion + MMR diversity reranking. Degrades to FTS-only if every embedding provider is unreachable. |
| `trajectory_search(query, k=3)` | Semantic search over past session summaries (task, outcome, skills used, reflection). Higher-signal than session_search — distilled lessons, not raw transcripts. |
| `skill_search(query, k=5)` | Semantic retrieval over available skills ranked by UCB1 bandit (exploration + exploitation). Records returned skills as candidates for stat tracking. Deprecated skills are excluded; near-duplicate hits are collapsed to siblings. |
| `skill_stats(name?)` | Read-only view of skill usage history: use count, success rate, status (draft/active/deprecated), last used. Omit `name` for a summary of all tracked skills. |
| `propose_skill(name, description, body, action="create")` | Create a new draft skill (`action="create"`) or rewrite an existing one (`action="edit"`). Skills start as `draft`, auto-promote to `active` after enough successful uses, and get `deprecated` if they chronically fail. Fuzzy patch matching handles indentation drift on edit. |
| `skill_rate(name, outcome)` | Manually rate a skill after use (`outcome ∈ {success, failure}`). Used to record cases where the hook cannot infer outcome automatically. |
| `reflect(content)` | Store a 2–4 sentence self-critique for the current session. Injected into the next iteration's prompt. With `reflection_scope="global"`, also embedded for cross-session recall. |
| `nano_status()` | Read-only snapshot of internal state: session ID, turns archived, salience score, nudge pending, reflection count, skill counts by lifecycle stage, DB size on disk. |
| `workflow_suggest(k=3)` | Cluster successful past trajectories by embedding similarity; surface recurring task patterns as workflow candidates. Prompts you to call `propose_skill` to codify them. Requires `workflow_induction.enabled = True`. |

---

## How it's wired

```
install(loop)
 ├── loop._extra_hooks.append(NanoHermesHook(config, loop))
 └── loop.tools.register(×10)

NanoHermesHook
 ├── BudgetedMemory        → wraps loop.context.memory (nanobot's MemoryStore)
 ├── SessionArchiver       → writes to <workspace>/nano_hermes/state.db
 ├── SkillIndexer          → reads loop.context.skills, writes skill_vec
 ├── TrajectoryWriter      → writes session summaries to trajectories + trajectories_vec
 ├── SkillUsageTracker     → skill stat accumulation, promotion/deprecation checks
 ├── ReflectionCoordinator → salience scoring, nudge injection, cross-session recall
 └── SessionCoordinator    → session boundary detection, trajectory finalization
```

**Per iteration:**

1. `before_iteration`:
   - Reset per-iteration counters.
   - Lazy-bootstrap a `sessions` row for this messages list.
   - On iteration 0: inject most similar past trajectory (if `inject_context=True`).
   - On iteration 0 with `reflection_scope="global"`: inject cross-session reflections relevant to the current task.
   - Inject any new reflections written since the last iteration (capped by `recent_limit`).
   - If a salience nudge is pending from last iteration, append the Reflexion nudge text.
   - If skill quality suggestions are pending (OPRO-inspired triggers), inject them.
2. LLM call (nanobot).
3. `before_execute_tools`: score tool-call bursts toward salience.
4. Tool execution (nanobot) — including any of our ten tools.
5. `after_iteration`:
   - Archive newly-appended messages: sync insert into `chunks` + schedule async embed.
   - Credit candidate skills with usage stats; run promotion/deprecation checks.
   - Add error + user-correction salience.
   - If cumulative score ≥ threshold, flip `_nudge_pending`.
   - On session boundary: write `ended_at`, finalize trajectory, record skill co-occurrences, prune internal dicts. Increment completed-session counter; if `rewrite_session_interval > 0` and the counter hits a multiple, schedule a background GEPA+rewriter evolution cycle.

**Skill lifecycle:**

```
propose_skill(action="create")  →  status="draft"
     ↓  (N successful uses, default N=3)
                                    status="active"
     ↓  (M uses with success_rate < 20%, default M=5)
                                    status="deprecated"  →  excluded from skill_search
     ↓  (re-propose)
propose_skill(action="create")  →  status="draft" (counters reset)
propose_skill(action="edit")    →  SKILL.md updated, counters preserved

Auto-evolution (if rewrite_session_interval > 0):
     every N sessions → GEPA pass (iterative mutation, if gepa_enabled=True)
                      → SkillForge rewriter (one-shot, severe failures)
     old text preserved in skill_versions table for diff history
```

**Data on disk:**

```
<workspace>/
├── memory/
│   ├── MEMORY.md
│   ├── USER.md
│   └── SOUL.md
├── skills/
│   └── <skill>/SKILL.md        ← propose_skill writes here
└── nano_hermes/
    └── state.db                ← sessions, chunks, chunks_fts, chunks_vec,
                                   skill_stats, skill_vec, skill_versions,
                                   skill_compositions, reflections,
                                   reflections_vec, trajectories, trajectories_vec
```

---

## Self-evolution

nano-hermes can automatically improve skills that are chronically failing, without any agent involvement. The pipeline has two stages and runs in the background at session boundaries.

### Trigger

Set `rewrite_session_interval` to a non-zero value (e.g. 5). After every N completed sessions, nano-hermes schedules a background evolution cycle. The cycle runs GEPA first, then the SkillForge rewriter — skills improved by GEPA are excluded from the rewriter in the same pass to avoid double-rewriting.

```json
{ "skill_stats": { "rewrite_session_interval": 5 } }
```

### Stage 1 — GEPA (off by default)

GEPA (Genetic-Pareto Prompt Evolution) is a gentler iterative pass. It targets active skills with failure rate ≥ `gepa_failure_threshold` (default 40%) and ≥ `gepa_min_uses` uses. For each candidate it runs up to `gepa_max_mutations` rounds of LLM mutation, scores each mutant on a Pareto frontier over (estimated improvement, token count), and promotes the best one via `propose_skill edit`.

Enable once you have ≥5 sessions of failure data:

```json
{
  "skill_stats": {
    "gepa_enabled": true,
    "gepa_failure_threshold": 0.4,
    "gepa_min_uses": 5,
    "gepa_max_mutations": 3
  }
}
```

### Stage 2 — SkillForge rewriter (always runs when triggered)

The rewriter targets skills with failure rate ≥ `rewrite_failure_threshold` (default 60%) and ≥ `rewrite_min_uses` uses. It gathers the `rewrite_context_chunks` most recent failed-session chunks for the skill, sends them to the LLM with an immutable judge prompt, safety-scans the output, and saves it as a draft via the normal `propose_skill edit` path. The original text is preserved in the `skill_versions` table for diff history.

The judge prompt is a module-level constant and cannot be overwritten by skill content — this prevents the metric-gaming failure mode where an optimized skill learns to game its own evaluator.

### Safety

All rewritten skill text passes through the same injection scanner (`skills/guard.py`) that `propose_skill` uses. A rewrite that contains prompt-injection patterns is logged and discarded; the original skill is unchanged.

---

## Verifying it's wired in

```python
hook = nano_hermes.install(loop)
for tool in [
    "memory_patch", "session_search", "trajectory_search",
    "skill_search", "skill_stats", "propose_skill", "skill_rate",
    "reflect", "nano_status", "workflow_suggest",
]:
    assert tool in loop.tools, f"missing: {tool}"
print(type(hook).__name__)   # NanoHermesHook
```

Or in the REPL after `nano-hermes agent`:
```
you> /tools
```
You should see all ten tools alongside nanobot's builtins.

---

## Troubleshooting

**`session_search` always returns `no matches (embedding unavailable: …)`.**
All three API keys are missing or invalid. Keep at least one working; FTS fallback still works without any key.

**`skill_search` returns `Error: cannot search skills — every embedding provider failed`.**
Same fix. Unlike `session_search`, `skill_search` has no FTS fallback (nanobot's alphabetical skill list is the fallback).

**Reflections don't appear in the next iteration.**
Check `hook.current_session_id is not None` — if you call `reflect` before any `before_iteration` has fired, the tool returns an error. The wrapper CLI and SDK path both trigger `before_iteration` automatically.

**`propose_skill` fails with "already exists".**
The skill is `active` or `draft`. Use `action="edit"` to update it, or wait for deprecation and re-create.

**`workflow_suggest` returns "disabled".**
Set `workflow_induction.enabled = True` in config. Requires at least `min_cluster_size` (default 3) successful session trajectories with overlapping task embeddings before any pattern surfaces.

**`memory_patch(action="distill")` returns "no recurring cross-session hubs found".**
Need ≥2 successful sessions (`outcome='ok'` in trajectories) with thematically overlapping content in `chunks`. Run more sessions and retry.

**`sqlite3.OperationalError: no such module: vec0`.**
```python
import sqlite_vec, sqlite3
conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
print("vec OK")
```
If this fails, `pip install sqlite-vec --no-binary :all:` to build from source.

**I want to see salience scores.**
```python
import logging
logging.getLogger("nano_hermes").setLevel(logging.DEBUG)
```
Every `after_iteration` logs `salience=X.X nudge=True/False`.

---

## Uninstall

```bash
uv tool install nanobot-ai --reinstall    # reinstalls without the plugin
```

The `state.db` under your workspace stays on disk — history is preserved if you re-enable later.
