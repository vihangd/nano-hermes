# nano-hermes

Self-evolving memory and skill extensions for [HKUDS/nanobot](https://github.com/HKUDS/nanobot).

Adds four agent-facing tools — `memory_patch`, `session_search`, `skill_search`, `reflect` — plus a lifecycle hook that archives turns into a searchable SQLite index, runs Reflexion-style self-critique on salience thresholds, and maintains a Voyager-style embedding index over nanobot's skill library.

Designed for low-resource hosts (Raspberry Pi) — no local embedding model, one SQLite file per workspace, hosted-embedding failover across DeepInfra, Together, and OpenRouter.

---

## Requirements

- **Python 3.11+** (tested on 3.14).
- **nanobot-ai** — the host framework.
- **One or more embedding API keys**: `DEEPINFRA_API_KEY`, `TOGETHER_API_KEY`, `OPENROUTER_API_KEY`. Failover walks the chain in that order; any one is enough to start.
- **bubblewrap** (optional) — only needed if you want sandboxed skill script execution, which nano-hermes delegates to nanobot's existing `bwrap` wrapper.

---

## Install

### On a Pi with `uv tool install nanobot-ai`

`uv tool` installs each CLI into its own isolated venv. nano-hermes has to live in the **same venv** as nanobot for the wrapper CLI to see it.

```bash
# One-shot install + plugin injection
uv tool install nanobot-ai --with-editable /path/to/nano-hermes

# Or if nanobot is already installed, reinstall with the plugin:
uv tool install nanobot-ai --with-editable /path/to/nano-hermes --reinstall
```

Swap `--with-editable` for `--with` if you don't plan to edit nano-hermes locally.

After install you get **two** commands on `PATH`:
- `nanobot` — unchanged, runs without nano-hermes.
- `nano-hermes` — same nanobot CLI, with nano-hermes auto-wired into every `AgentLoop`.

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
pytest -v                            # 33 tests, expect all green
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

That's it. The wrapper monkey-patches `AgentLoop.__init__` at import time and calls `nano_hermes.install(loop)` on every instance nanobot constructs. Your existing `~/.nanobot/config.json` is respected.

### Alternative — Python SDK (embedded use)

For programmatic use or scripts:

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

### Via environment variables (primary)

```bash
DEEPINFRA_API_KEY=sk-...      # embedding chain primary
TOGETHER_API_KEY=...          # failover
OPENROUTER_API_KEY=...        # failover
NANO_HERMES_ROOT=~/.nano-hermes  # optional — only used if you call functions from paths.py directly
```

The state database lives under `<workspace>/nano_hermes/state.db`, where `workspace` is nanobot's workspace (usually `~/.nanobot/workspace` or whatever you set in nanobot's config).

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
    "retrieval": {
        "final_k": 12,             # default 8 — hits returned by session_search
    },
    "embedding": {
        "target_dims": 512,        # default 512, truncated from bge-m3's native 1024
    },
    "trajectory_retention_days": 45,
})
```

### Embedding model override

Defaults to `BAAI/bge-m3` truncated to 512 dims. To switch providers or models:

```python
config={
    "embedding": {
        "model": "BAAI/bge-small-en-v1.5",   # English-only, 384 dims
        "native_dims": 384,
        "target_dims": 384,
        "chain": [
            {"provider": "deepinfra", "api_key_env": "DEEPINFRA_API_KEY"},
            {"provider": "together",  "api_key_env": "TOGETHER_API_KEY"},
        ],
    }
}
```

---

## What the agent gets

Four tools land on `loop.tools`:

| Tool | What it does |
|---|---|
| `memory_patch(slot, action, content?, needle?, replacement?)` | Edit `MEMORY.md` / `USER.md` / `SOUL.md` under nanobot's workspace. Enforces char budgets and reports exact shortfall on overflow. `slot ∈ {memory, user, soul}`, `action ∈ {add, replace, remove}`. |
| `session_search(query, k=8)` | Hybrid FTS5 + embedding search over archived turn chunks. RRF fusion. Degrades to FTS-only if every embedding provider is unreachable. |
| `skill_search(query, k=5)` | Semantic retrieval over available skills (ranked by name+description embedding). Complements nanobot's alphabetical skill list. |
| `reflect(content)` | Store a 2–4 sentence self-critique scoped to the current session. Injected into the next iteration's prompt. |

Plus a lifecycle hook that runs on every iteration — see below.

---

## How it's wired

```
install(loop)
 ├── loop._extra_hooks.append(NanoHermesHook(config, loop))
 └── loop.tools.register(×4)

NanoHermesHook
 ├── BudgetedMemory  → wraps loop.context.memory (nanobot's MemoryStore)
 ├── SessionArchiver → writes to <workspace>/nano_hermes/state.db
 ├── SkillIndexer    → reads loop.context.skills, writes skill_vec
 └── current_session_id / salience counters / reflection bookkeeping
```

**Per iteration:**

1. `before_iteration`:
   - Reset counters.
   - Lazy-bootstrap a `sessions` row for this messages list if it's the first time we see it.
   - Fetch any reflections written since the last iteration for this session; append them to `context.messages` as a system message.
   - If a salience nudge is pending from last iteration, append the Reflexion nudge text.
2. LLM call (nanobot).
3. `before_execute_tools`: score tool-call bursts toward salience.
4. Tool execution (nanobot) — including any of our four tools.
5. `after_iteration`:
   - Archive any newly-appended messages: sync insert into `chunks` (FTS5 stays current via trigger) + schedule async embed to `chunks_vec`.
   - Add error + user-correction salience.
   - If cumulative score ≥ threshold, flip `_nudge_pending` for next iteration.

**Skills path:**

Nanobot's `SkillsLoader` still runs — it discovers `workspace/skills/` + builtin skills, builds an XML summary, and injects it into the system prompt. Nothing we do touches that. `skill_search` is *additive*: on call, it opens the embedding chain, refreshes the index (skips unchanged skills via content-hash), embeds the query, runs sqlite-vec top-k, returns `[distance] name — description (path)` lines. The agent follows the `path` with `read_file` to load the full body.

**Data on disk:**

```
<workspace>/
├── memory/                  # nanobot's — we write MEMORY.md etc. through its MemoryStore
│   ├── MEMORY.md
│   ├── USER.md
│   └── SOUL.md
├── skills/                  # nanobot's — your skill library
│   └── <skill>/SKILL.md
└── nano_hermes/
    └── state.db             # our SQLite: sessions/chunks/chunks_fts/chunks_vec/
                              # skill_stats/skill_vec/reflections/trajectories
```

---

## Verifying it's wired in

```bash
nano-hermes agent
```
In the REPL, check the available tools:
```
you> /tools
```
You should see `memory_patch`, `session_search`, `skill_search`, `reflect` alongside nanobot's builtins (`read_file`, `write_file`, `shell`, `web_search`, …).

Or from Python:
```python
hook = nano_hermes.install(loop)
print("memory_patch" in loop.tools)    # True
print("session_search" in loop.tools)  # True
print("skill_search" in loop.tools)    # True
print("reflect" in loop.tools)         # True
print(type(hook).__name__)             # NanoHermesHook
```

Ask the agent something that forces `session_search`:
```
you> what did we decide about the trip to Reykjavik last week?
```
If the current session (or past sessions within the 45-day retention window) mention Reykjavik, you'll see the agent call `session_search` and get hits.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'nano_hermes'` when running `nano-hermes agent`.**
The plugin isn't in nanobot's venv. Re-run the install with `--with-editable` pointing at your nano-hermes checkout. If you installed via pip instead of uv tool, check that you're in the venv where nanobot lives.

**`sqlite3.OperationalError: no such module: vec0`.**
`sqlite-vec` didn't load. Make sure you're on Python 3.11+ and that `pip install sqlite-vec` pulled a wheel (or built from source). Verify with:
```python
import sqlite_vec, sqlite3
conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
print("vec OK")
```

**`session_search` always returns `no matches (embedding unavailable: …)`.**
All three API keys are missing or invalid. Unset the broken ones so the chain walks faster; keep at least one working.

**`skill_search` returns `Error: cannot search skills — every embedding provider failed`.**
Same fix as above. Unlike `session_search`, `skill_search` can't fall back to FTS because skill names/descriptions aren't indexed in `chunks_fts`.

**Reflections don't appear in the next iteration.**
Check `hook.current_session_id is not None` — if you call `reflect` before any `before_iteration` has fired, the tool returns an error explaining there's no session row yet. The wrapper CLI and the SDK path both trigger `before_iteration` automatically.

**I want to see what salience score I'm at.**
Set log level to DEBUG:
```python
import logging
logging.getLogger("nano_hermes").setLevel(logging.DEBUG)
```
Every `after_iteration` will log `salience=X.X nudge=True/False`.

---

## Uninstall

If you installed via `uv tool`:
```bash
uv tool install nanobot-ai --reinstall    # reinstalls without the plugin
```

If you installed via `pip install -e .` in a venv:
```bash
pip uninstall nano-hermes
```

Either way, the `state.db` file under your nanobot workspace stays on disk until you delete it — history is preserved even if you re-enable the plugin later.
