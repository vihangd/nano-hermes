# Pre-submission quality checklist

Read this right before you call `propose_skill`. It's the pre-flight gate: either the skill passes every item below, or it goes back to the drafting board.

## 1. Pre-submission checklist

Walk through each item. Any "no" is a blocker.

- [ ] **Name**: hyphen-lowercase, ≤64 chars, matches `^[a-z0-9][a-z0-9_-]{0,63}$`. Task-driven, not tool-driven. No version suffixes.
- [ ] **Description**: starts with "Use when…" or an equivalent trigger phrase. Names the key domain noun. ≤200 chars. Specific enough to rule out irrelevant queries.
- [ ] **Duplicate check**: you called `skill_search` with the same intent and did not find an existing `active` skill. (Or you are proposing an edit of one via `action="edit"`.)
- [ ] **Body structure**: follows the template from `authoring-guide.md` — Overview / When to use / Procedure / Examples / Edge cases / (Guidelines).
- [ ] **Procedure is deterministic**: each step is testable. No "figure out" placeholders. Scripts handle the parts that need determinism.
- [ ] **No local context leakage**: no absolute paths, no hard-coded usernames, no host-specific commands. A fresh agent in a different environment could run this.
- [ ] **Scripts documented**: every companion script has its invocation documented in the body's Procedure section. Exit codes and error handling are explicit.
- [ ] **Worked example**: body contains at least one concrete user intent → action example.
- [ ] **Size**: total (body + files) is under ~256 KiB. If it's within 10% of the cap, trim before submitting.
- [ ] **Fresh-agent test**: you read your own SKILL.md as if you were a new agent and could execute it without inventing facts.
- [ ] **You plan to rate it**: you understand that `propose_skill` is step 1 of 2, and you will call `skill_rate` after first real use.

If every box is checked, submit.

## 2. Anti-patterns

Stop and refactor if you notice any of these while drafting.

| Anti-pattern | Symptom | Fix |
|---|---|---|
| Description only says *what*, not *when* | Poor semantic retrieval — other agents can't find it | Lead with "Use when…"; name the trigger explicitly |
| Body embeds local paths or usernames | Skill fails for other sessions / users | Cut them or move to `references/local-setup.md` |
| Script shipped untested | Agent runs it, it crashes, skill deprecates | Walk through the script's logic mentally or test it in the current session before including |
| Body over 500 lines | Agent can't hold it in context alongside the task | Push detail into `references/*.md` or split into two skills |
| Two skills glued together | Name doesn't fit one sentence | Split and propose both separately |
| Duplicate of existing active skill | Pollutes search results, dilutes ratings | `skill_search` first; use `action="edit"` to fix an existing skill |
| `propose_skill` called, never rated | Stuck at draft forever, never promotes | Rate after first real use — success OR failure |
| Scripts with decorative logging to stdout | Agent has to parse around the noise | Keep stdout structured; send logs to stderr |
| Version suffix on name (`fetch-arxiv-v2`) | Orphans the old skill's counters | Edit, don't re-create; `action="edit"` preserves stats |
| Description that promises capabilities the body doesn't deliver | Skill retrieves but fails when applied | Audit: description must match body exactly |
| Body duplicates information that's already in nanobot's built-in tools | Skill is redundant wrapping | Point to the built-in tool, don't re-document it |

## 3. Lifecycle integration — draft→active is the real completion gate

`propose_skill` creates a draft. A draft is searchable but unvalidated. Promotion to `active` only happens when the skill's empirical success rate earns it.

**Promotion:** After `skill_rate(name, outcome="success")` is called `promotion_threshold` times (default 3), nano-hermes flips `status` to `active`. Failure ratings do not count toward promotion.

**Deprecation:** After `deprecation_min_uses` uses (default 5), nano-hermes checks the success rate. If it's under `deprecation_max_success_rate` (default 20%), the skill auto-deprecates and disappears from search. The skill isn't deleted — you can re-propose a deprecated skill with fresh content and reset counters via `action="create"`.

**Why this matters:**

- A skill that is proposed but never rated stays at 0/0 forever. It's technically searchable (drafts are), but it has no evidence it works. Treat unrated drafts as debt.
- Rating with `"failure"` is not a punishment — it's information. A skill that fails once might still earn promotion if it later succeeds three times. A skill that fails chronically is correctly removed.
- Edit existing skills via `action="edit"` to preserve counters. Re-creating with the same name resets everything — only do this if the skill genuinely needs a clean slate.

**The loop you should maintain:**

1. `propose_skill` → creates draft.
2. Use the skill in a real task.
3. `skill_rate` with honest outcome.
4. Repeat 2–3 until the skill either promotes (3 successes) or deprecates (5 uses, <20% success).

Without step 3, the skill never leaves draft. Step 3 is not optional.

## 4. When NOT to propose at all

Be suspicious of your urge to propose. Most "I should make a skill for this" instincts are wrong.

- **One-off task.** If you'll never need this procedure again, just do the work. Don't pollute the skill library.
- **Fits in a memory entry.** If the knowledge is a fact (≤200 chars, not a procedure), call `remember` instead. Memory is for facts; skills are for procedures.
- **Already covered.** If `skill_search` finds an `active` skill that matches the intent, use it. If it needs a bug fix, propose an edit.
- **User-asked but single-use.** If the user says "make a skill for this" but the skill would only ever run in this one conversation, resist. Save the context as a memory entry instead.
- **You're proposing to please the user, not to persist value.** A skill you don't believe in will never get rated `"success"`. It'll sit in draft forever, wasting retrieval budget.

Propose skills you'd actually want a future version of yourself to use.
