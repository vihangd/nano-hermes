---
name: skill-creator
description: Create a new reusable skill via propose_skill. Integrates with the nano-hermes lifecycle (draft → active → deprecated) and runs security checks automatically.
---

# Creating a skill

Call `propose_skill` with the following parameters:

- **`name`** — hyphen-lowercase identifier (e.g. `plot-time-series`, `fetch-arxiv`). This becomes the directory name under `workspace/skills/`.
- **`description`** — one-line summary of what the skill does and when to use it. This is embedded for semantic search — precision matters.
- **`body`** — the Markdown content of the skill: when to use it, how to invoke it, step-by-step workflow, examples, common failure modes.
- **`files`** *(optional)* — companion files alongside SKILL.md. Each entry is `{"path": "scripts/foo.py", "content": "..."}`. Allowed subdirectories:
  - `scripts/` — executable helpers (Python `.py`, Node `.js`/`.mjs`/`.ts`, shell `.sh`/`.bash`)
  - `references/` — Markdown reference docs, API specs, cheat sheets
  - `assets/` — static data, templates, example files

## Lifecycle

1. A newly proposed skill starts as **draft**. It is searchable but unvalidated.
2. After each session where you use the skill, call `skill_rate` with `"success"` or `"failure"`.
3. After enough successes (default: 3), the skill promotes to **active**.
4. Skills with chronically low success rates are automatically **deprecated** and hidden from search.

## When NOT to create a skill

- For one-off tasks — just complete the work without persisting it.
- For knowledge that fits in a memory entry — use `remember` instead.
- If an `active` skill already covers the use case (check with `skill_search` first).

## Example: skill with a helper script

```
propose_skill(
  name="run-pytest",
  description="Run pytest in the current repo with sane defaults and summarise failures.",
  body="## Usage\nCall scripts/run.sh from the repo root...",
  files=[
    {
      "path": "scripts/run.sh",
      "content": "#!/usr/bin/env bash\nset -euo pipefail\npytest -v --tb=short \"$@\"\n"
    }
  ]
)
```

## Notes

- Never write skill files directly with shell commands or file tools — doing so bypasses nano-hermes and the skill will start as `draft` only after the indexer discovers it on the next `skill_search`.
- The `edit` action updates an existing draft or active skill without resetting usage counters. Read the current SKILL.md with `read_file` before editing.
- Deprecated skills can be re-proposed with `action="create"` to start fresh with reset counters.
