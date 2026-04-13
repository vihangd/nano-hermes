# Skill authoring guide — the craft

Read this when you are actively authoring a skill via `propose_skill`. The SKILL.md router tells you *when* and *how* at a high level; this file covers the craft details that determine whether the skill survives the promotion gate and retrieves well in future sessions.

## 1. Description craft — the semantic-search contract

The name + description pair is the **only** text embedded for semantic search in nano-hermes. Body content, references, scripts, and assets are *not* indexed — they never affect retrieval ranking. This means the description is disproportionately important. Spend real effort on it.

**Rules:**

- **Lead with the trigger.** Start with "Use when…" or an equivalent phrase that describes the user intent the skill solves. Agents search by intent, not by tool name.
- **Name the domain noun.** If the skill is about pandas DataFrames, the word "DataFrame" must appear. If it's about arxiv, "arxiv" must appear. Vague nouns kill retrieval.
- **Be specific about constraints.** "Renders via matplotlib, saves PNG" beats "produces a chart". Constraints rule the skill in for relevant queries and rule it out for irrelevant ones.
- **Length**: target 120–200 characters. Hard cap 1024 (enforced by nanobot's `quick_validate.py`). Search result display may truncate around 160 chars, so put the trigger and domain noun early.

**Good vs bad:**

| Bad | Good |
|---|---|
| "Plots data." | "Use when the user asks for a time-series plot from a pandas DataFrame. Renders via matplotlib, saves PNG, handles missing timestamps by forward-fill." |
| "Helper for arxiv." | "Use when the user asks to fetch arxiv papers by ID or search query. Parses the arxiv Atom feed, returns title/authors/abstract/pdf URL." |
| "Runs tests." | "Use when the user asks to run pytest in a Python repo. Executes with verbose output and short traceback, summarises failing test names and error categories." |

Notice the pattern: trigger phrase ("Use when…"), domain noun, concrete behaviour. Each description names what the skill does *specifically*, so a future semantic-search query for "plot time series", "arxiv paper fetch", or "run pytest failures" lands on it.

## 2. Body section template

Use this structure for the `body` parameter. Drop any section you can't fill honestly — empty sections are worse than missing ones.

```
## Overview           — one paragraph, what the skill does and why it exists
## When to use        — trigger phrases, example user intents
## Procedure          — numbered steps, deterministic, testable
## Examples           — 1-2 worked examples (short, verbatim user prompts + your action)
## Edge cases         — known failure modes, how to recover
## Guidelines         — opinions the agent should carry forward (optional)
```

The body is NOT embedded for search, but it *is* what you read when you load the skill to apply it. So optimize for fast scanning by a future agent who has already decided this skill is relevant and now needs to execute it.

**Procedure discipline.** Number the steps. Make each step testable. If a step says "figure out X", it's not a procedure — it's a wish. Either ship a script that figures out X, or break the step down until each sub-step is deterministic.

## 3. Progressive disclosure — what goes where

Keep SKILL.md lean. Push anything the agent won't always need into `references/`.

- **SKILL.md body**: The router. If the whole skill fits in under ~300 lines, put everything here.
- **`references/*.md`**: Conditional knowledge — API specs, cheat sheets, decision tables, longer examples, edge-case catalogs. The agent reads these via `read_file` using the path sibling to SKILL.md. Only load when needed.
- **`scripts/*`**: Deterministic helpers the agent should *run*, not re-derive. Python (`.py`), Node (`.js`, `.mjs`, `.ts`), shell (`.sh`, `.bash`) — all fine. nano-hermes scans `scripts/` with relaxed rules so legitimate language constructs like `eval(`, `exec(`, `__import__(` are allowed (the destructive-shell and exfiltration checks still apply).
- **`assets/*`**: Static data, templates, fixtures. Keep under the 256 KiB total budget.

**Rules of thumb:**
- If the body refers to a section of content more than once, split it into a reference.
- If a script is longer than the body that invokes it, the script is probably the real skill.
- If an example is over 30 lines, move it to `references/examples.md`.

## 4. When to ship a script vs instruct inline

**Ship a script when:**
- The step is a deterministic sequence that doesn't vary by context.
- Correctness is critical — parsing, validation, formatting, cryptography.
- The agent would otherwise re-derive the same code every session.
- The output is easier to consume as parseable text than to describe in prose.

**Instruct inline when:**
- The step is exploratory and varies by context.
- The agent's reasoning genuinely improves the outcome.
- The shape of the input or output isn't fixed.

**Script quality bar:**
- Handle failure explicitly. Shell: `set -euo pipefail`. Python: non-zero exit on error with a clear message. Node: reject unhandled promises.
- Accept clear inputs via argv or stdin — document the invocation in the body.
- Produce parseable output — one line per record, or JSON. Don't emit decorative logging to stdout that the agent then has to parse around.
- Document the exact invocation in SKILL.md's Procedure section. Don't make the agent reverse-engineer your CLI.

## 5. Naming and scope

- Names are hyphen-lowercase, ≤64 chars, matching `^[a-z0-9][a-z0-9_-]{0,63}$`.
- Prefer **task-driven** names (`fetch-arxiv`, `plot-time-series`, `run-pytest`) over **tool-driven** names (`pandas-helper`, `api-wrapper`, `utils`).
- If you can't summarize the skill in one sentence, it's probably two skills. Split.
- Don't use version suffixes (`-v2`, `-new`). If you need to change the skill, use `propose_skill(action="edit", ...)` — it preserves the success counters that drive promotion.

## 6. Fresh-agent usability test

Before submitting, imagine a fresh nano-hermes agent in a completely different session — same tools, same nano-hermes, but no memory of this conversation. Read your SKILL.md through their eyes. Ask:

- **Can they execute the procedure without inventing facts?** If the body says "then run the usual build command", they can't. Spell it out or ship a script.
- **Does the body assume anything about the local environment?** Absolute paths, specific username, host-specific binaries. If yes, cut them or move them to a `references/local-setup.md` stub that the user fills in.
- **Would they understand *when* to apply this?** If the description is vague, they won't retrieve it at all. If the body's "When to use" section is vague, they'll apply it wrong.

If you fail the fresh-agent test, the skill isn't ready — don't submit it as active, even if the lifecycle would let you.

## 7. Size discipline

- **SKILL.md body**: target ≤300 lines, hard cap 500. Beyond that, push to references.
- **Total bytes** (body + all companion files) ≤ 256 KiB. `propose_skill` enforces this; over-size calls are rejected before anything lands on disk.
- If you can't stay under the cap, you're probably writing a library rather than a skill. Split it into two skills, or extract a script that does the heavy lifting.

## 8. Editing, not re-creating

When fixing a bug in an existing skill, use `propose_skill(action="edit", ...)`. It preserves the `use_count`, `success_count`, and `last_used_at` counters that drive promotion. Re-creating with `action="create"` resets them to zero — the skill has to earn its trust all over again.

You can pass `files=[...]` in edit mode to add or overwrite companion files, and `delete_files=[...]` to remove them. Edit is atomic: scan succeeds → all files land, scan fails → nothing changes.

## 9. What you get for free from nano-hermes

You don't need to worry about:

- **Scaffolding directories** — `propose_skill` creates `{workspace}/skills/{name}/` and subdirectories for you.
- **Validation** — paths, filenames, and content are scanned before any write; the orphaned-directory guard prevents silent adoption of shell-created skills.
- **Rollback** — if any part of the write sequence fails in create mode, nano-hermes `rmtree`s the partial directory so you can retry cleanly. Edit mode preserves your existing content.
- **Search indexing** — the next `skill_search` call picks up your new skill automatically; no manual refresh needed.
- **Lifecycle tracking** — `skill_rate` drives promotion and deprecation; you just report outcomes.

Don't recreate any of this in your skill. Trust the platform.
