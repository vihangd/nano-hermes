---
name: skill-creator
description: Read this first when you're about to author a new skill via propose_skill. Teaches description craft, body structure, progressive disclosure, scripting decisions, and the draft→active rating loop.
---

# Creating a skill

You author reusable skills via the `propose_skill` tool. This skill is the router — it tells you when to author, what shape the output must take, and where to read for the craft details. Treat `propose_skill` as the *write*, and treat `skill_rate` as the *completion gate*. A skill that is proposed but never rated stays draft forever.

## When to use this skill

Use this skill when you are about to call `propose_skill` because you have a reusable procedure worth persisting across sessions — a reliable way to call an API, a multi-step data-processing recipe, a debugging checklist.

**Do NOT propose a skill when:**
- The task is one-off. Just do the work.
- The knowledge fits in ≤200 chars and is factual, not procedural. Use `remember` instead.
- An `active` skill already covers the use case. Search first with `skill_search`; propose an `edit` if it needs a fix.
- The user *asked* you to, but the skill would only run once. Resist — save the context as a memory.

## 5-step authoring workflow

1. **Capture intent and scope.** What problem is this solving? Is the procedure deterministic and reusable, or context-dependent and one-off? One sentence. If you can't fit the purpose in one sentence, it's probably two skills.
2. **Search first.** Call `skill_search` with the intent. If a matching `active` skill exists, use it; propose an `edit` if broken. Do not duplicate.
3. **Draft in your head.** Work out the description, the body section outline, and the companion file list (scripts, references, assets). Skills are atomic — `propose_skill` takes everything in one call, so plan the whole thing before writing.
4. **Read the references.** Read `references/authoring-guide.md` for the body template, description craft, and progressive-disclosure choices. Read `references/quality-checklist.md` for the pre-submission checklist and anti-patterns.
5. **Submit, then rate.** Call `propose_skill(name, description, body, files)`. After you actually *use* the skill in a later turn and observe the outcome, call `skill_rate(name, outcome="success"|"failure")`. Promotion to `active` requires 3 successful ratings by default; without them the skill stays draft and never accumulates trust.

## STOP — red flags

Abort or restructure if any of these apply:

- **Description says only *what*, not *when*.** Leads to poor retrieval. Lead with "Use when…".
- **Body embeds absolute paths, usernames, or host-specific commands.** The skill fails in other sessions. Move to `references/local-setup.md` or remove.
- **You're shipping a script you haven't thought through.** Script behavior must be deterministic and documented in the body.
- **You're reimplementing a library in the script.** If a published package already solves the hard part, the wrapper should invoke it via `uvx` / `bunx` / `npx` — not re-derive its logic. See `references/authoring-guide.md` §4 for the wrapper pattern.
- **Body is over 500 lines.** Split into `references/` files or into two skills.
- **The skill is really two skills glued together.** Name doesn't fit one sentence. Split.
- **You plan to propose and never rate.** The draft→active gate exists for a reason. Rate after first real use.

## Minimal example

This example demonstrates the **wrapper pattern**: the complex work (running tests, parsing output) is delegated to pytest via `uvx`; the wrapper script is thin and only adapts the invocation for the skill's needs.

```python
propose_skill(
  name="run-pytest",
  description=(
    "Use when the user asks to run pytest in the current Python repo. "
    "Executes pytest via uvx (no pre-install needed), verbose output with "
    "short traceback, summarises failing test names and error categories."
  ),
  body="""
## Overview
Run pytest via `uvx` with sane defaults and summarise failures. Uses
uvx so the skill works in any Python repo without assuming a pre-
installed pytest or a specific .venv layout.

## When to use
User asks to run tests, run pytest, check test failures, or rerun
the last failures in a Python repo.

## Procedure
1. Check `pyproject.toml`, `pytest.ini`, or a `tests/` directory
   exists at the cwd. If none, ask the user to confirm.
2. Run `bash scripts/run.sh` (optionally with extra pytest args, e.g.
   `-k name` or `--lf`). The script invokes `uvx pytest@8.3.0`
   internally — pinned for reproducibility.
3. Summarise the FAILED/ERROR lines for the user. For many failures,
   offer to rerun the first N with `--tb=long`.

## Examples
- User: "run the tests"
  → `bash scripts/run.sh`, summarise
- User: "rerun the last failures"
  → `bash scripts/run.sh --lf`

## Edge cases
- No pyproject.toml and no tests/ dir → ask for the test command
- First run is slow → uvx is downloading pytest; subsequent runs are cached
- Offline with no uvx cache → fall back to a locally-installed pytest
  if available, otherwise report the network requirement

## Guidelines
- Don't reimplement pytest output parsing — let pytest's own `-v` and
  `--tb=short` do the formatting; the wrapper only filters.
""",
  files=[
    {
      "path": "scripts/run.sh",
      "content": (
        "#!/usr/bin/env bash\n"
        "# Thin wrapper around uvx pytest — pinned version for reproducibility\n"
        "set -euo pipefail\n"
        "uvx pytest@8.3.0 -v --tb=short \"$@\"\n"
      ),
    },
    {
      "path": "references/flags.md",
      "content": "# Pytest flags\n\n- `-k EXPR` — run tests matching EXPR\n- `--lf` — last-failed\n- `-x` — stop on first fail\n- `-n auto` — parallel (requires pytest-xdist)\n",
    },
  ],
)
```

The script is 4 lines — all the test-running complexity lives in pytest itself, invoked via `uvx`. After using this skill for the first time in a real session, call `skill_rate(name="run-pytest", outcome="success")` (or `"failure"`).

## Pointers

- `references/authoring-guide.md` — the craft: body template, description writing, progressive disclosure, when to ship a script, naming, fresh-agent usability test, size discipline.
- `references/quality-checklist.md` — the gate: pre-submission checklist, anti-patterns table, lifecycle integration, when NOT to propose at all.

Read both before calling `propose_skill` for the first time in a session.
