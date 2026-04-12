# nano-hermes

Self-evolving memory, skill lifecycle, and Reflexion-based self-improvement extensions for [HKUDS/nanobot](https://github.com/HKUDS/nanobot).

Adds eight agent-facing tools plus a lifecycle hook that archives turns into a searchable SQLite index, runs Reflexion-style self-critique, maintains a Voyager-style embedding index over nanobot's skill library, and supports a two-phase skill promotion system (draft → active → deprecated).

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

### For developing nano-hermes itself

```bash
cd /path/to/nano-hermes
python3 -m venv .venv && source .venv/bin/activate
pip install -e /path/to/nanobot      # local nanobot checkout
pip install -e '.[dev]'              # nano-hermes + pytest + sqlite-vec
pytest -v                            # 97 tests, expect all green
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

### Via install() config

```python
nano_hermes.install(loop, config={
    "memory": {
        "memory_md_chars": 3000,   # default 2200
        "user_md_chars": 2000,     # default 1375
        "soul_md_chars": 2000,     # default 1500
    },
    "reflection": {
        "threshold": 3.0,          # default 5.0 — lower = more nudges
        "recent_limit": 8,         # default 5 — max reflections injected per iter
    },
    "reflection_scope": "global",  # "session" (default) or "global" for cross-session recall
    "retrieval": {
        "final_k": 12,             # default 8 — hits returned by session_search
    },
    "skill_stats": {
        "promotion_threshold": 5,         # default 3 — successful uses to promote draft
        "deprecation_min_uses": 10,       # default 5 — uses before deprecation check
        "deprecation_max_success_rate": 0.1,  # default 0.2 — below this → deprecated
    },
    "trajectory": {
        "inject_context": True,           # default False — inject similar past trajectories
        "inject_min_similarity": 0.8,     # default 0.75 — similarity threshold
    },
    "trajectory_retention_days": 45,
})
```

---

## What the agent gets

Eight tools land on `loop.tools`:

| Tool | What it does |
|---|---|
| `memory_patch(slot, action, ...)` | Edit `MEMORY.md` / `USER.md` / `SOUL.md`. Enforces char budgets; blocks prompt injection and invisible unicode. `slot ∈ {memory, user, soul}`, `action ∈ {add, replace, remove}`. |
| `session_search(query, k=8)` | Hybrid FTS5 + embedding search over archived turn chunks. RRF fusion. Degrades to FTS-only if every embedding provider is unreachable. |
| `trajectory_search(query, k=3)` | Semantic search over past session summaries (task, outcome, skills used, reflection). Higher-signal than session_search — distilled lessons, not raw transcripts. |
| `skill_search(query, k=5)` | Semantic retrieval over available skills ranked by embedding similarity. Records returned skills as candidates for stat tracking. Deprecated skills are excluded. |
| `skill_stats(name?)` | Read-only view of skill usage history: use count, success rate, status (draft/active/deprecated), last used. Omit `name` for a summary of all tracked skills. |
| `propose_skill(name, description, body, action="create")` | Create a new draft skill (`action="create"`) or rewrite an existing one (`action="edit"`). Skills start as `draft`, auto-promote to `active` after enough successful uses, and get `deprecated` if they chronically fail. |
| `reflect(content)` | Store a 2–4 sentence self-critique for the current session. Injected into the next iteration's prompt. With `reflection_scope="global"`, also embedded for cross-session recall. |
| `nano_status()` | Read-only snapshot of internal state: session ID, turns archived, salience score, nudge pending, reflection count, skill counts by lifecycle stage, DB size on disk. |

---

## How it's wired

```
install(loop)
 ├── loop._extra_hooks.append(NanoHermesHook(config, loop))
 └── loop.tools.register(×8)

NanoHermesHook
 ├── BudgetedMemory    → wraps loop.context.memory (nanobot's MemoryStore)
 ├── SessionArchiver   → writes to <workspace>/nano_hermes/state.db
 ├── SkillIndexer      → reads loop.context.skills, writes skill_vec
 ├── TrajectoryWriter  → writes session summaries to trajectories + trajectories_vec
 └── salience counters / reflection bookkeeping / skill stat tracking
```

**Per iteration:**

1. `before_iteration`:
   - Reset per-iteration counters.
   - Lazy-bootstrap a `sessions` row for this messages list.
   - On iteration 0: inject most similar past trajectory (if `inject_context=True`).
   - On iteration 0 with `reflection_scope="global"`: inject cross-session reflections relevant to the current task.
   - Inject any new reflections written since the last iteration (capped by `recent_limit`).
   - If a salience nudge is pending from last iteration, append the Reflexion nudge text.
2. LLM call (nanobot).
3. `before_execute_tools`: score tool-call bursts toward salience.
4. Tool execution (nanobot) — including any of our seven tools.
5. `after_iteration`:
   - Archive newly-appended messages: sync insert into `chunks` + schedule async embed.
   - Credit candidate skills with usage stats; run promotion/deprecation checks.
   - Add error + user-correction salience.
   - If cumulative score ≥ threshold, flip `_nudge_pending`.
   - On session boundary: write `ended_at`, finalize trajectory, embed task text.

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
                                   skill_stats, skill_vec, reflections,
                                   reflections_vec, trajectories, trajectories_vec
```

---

## Verifying it's wired in

```python
hook = nano_hermes.install(loop)
for tool in ["memory_patch", "session_search", "trajectory_search",
             "skill_search", "skill_stats", "propose_skill", "reflect",
             "nano_status"]:
    assert tool in loop.tools, f"missing: {tool}"
print(type(hook).__name__)   # NanoHermesHook
```

Or in the REPL after `nano-hermes agent`:
```
you> /tools
```
You should see all seven tools alongside nanobot's builtins.

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
