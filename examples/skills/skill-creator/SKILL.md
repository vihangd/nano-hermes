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
- **Body is over 500 lines.** Split into `references/` files or into two skills.
- **The skill is really two skills glued together.** Name doesn't fit one sentence. Split.
- **You plan to propose and never rate.** The draft→active gate exists for a reason. Rate after first real use.

## Minimal example

```python
propose_skill(
  name="run-pytest",
  description=(
    "Use when the user asks to run pytest in the current Python repo. "
    "Executes with verbose output and short traceback, summarises failing "
    "test names and error categories, handles --last-failed reruns."
  ),
  body="""
## Overview
Run pytest with sane defaults and summarise failures in a way that
helps the user fix them without scrolling through raw output.

## When to use
User asks to run tests, run pytest, check test failures, or rerun
the last failures in a Python repo.

## Procedure
1. Check `pyproject.toml` or `pytest.ini` exists. If not, ask the
   user to confirm the test command.
2. Run `scripts/run.sh` from the repo root. See references/flags.md
   if the user needs non-default flags.
3. Parse the output, group failures by test file, summarise each.
4. If there are many failures, offer to rerun the first N with --tb=long.

## Examples
- User: "run the tests"
  → call scripts/run.sh, summarise
- User: "rerun the last failures"
  → call scripts/run.sh with --lf

## Edge cases
- No pyproject.toml → ask for the test runner command
- pytest not installed → suggest `.venv/bin/pip install pytest`
""",
  files=[
    {
      "path": "scripts/run.sh",
      "content": (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ".venv/bin/pytest -v --tb=short \"$@\"\n"
      ),
    },
    {
      "path": "references/flags.md",
      "content": "# Pytest flags\n\n- `-k EXPR` — run tests matching EXPR\n- `--lf` — last-failed\n- `-x` — stop on first fail\n",
    },
  ],
)
```

After running this skill for the first time in a real session, call `skill_rate(name="run-pytest", outcome="success")` (or `"failure"`).

## Pointers

- `references/authoring-guide.md` — the craft: body template, description writing, progressive disclosure, when to ship a script, naming, fresh-agent usability test, size discipline.
- `references/quality-checklist.md` — the gate: pre-submission checklist, anti-patterns table, lifecycle integration, when NOT to propose at all.

Read both before calling `propose_skill` for the first time in a session.
