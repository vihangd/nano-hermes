---
name: duckduckgo-search
description: Web search via DuckDuckGo (no API key required)
---

# Web Search via DuckDuckGo

Use this skill when you need information from the public internet and
don't have a more specific tool for the source. DuckDuckGo works without
any credentials, so it's the lowest-friction option.

## When to reach for it

- Current events or anything that postdates your training.
- Public facts you're not confident about (spelling, dates, versions).
- Finding an authoritative source that you can then read in full with
  `web_fetch`.
- Sanity-checking a claim before acting on it.

Skip it when:
- The user wants your opinion or reasoning, not fresh facts.
- You already know the answer at high confidence.
- A more targeted tool exists: GitHub for issues, arXiv for papers,
  documentation sites via `web_fetch` if you know the URL.

## How to use it

Call nanobot's built-in `web_search` tool with `provider: "duckduckgo"`:

```
web_search(query="…", provider="duckduckgo", max_results=5)
```

Query-writing rules, in descending importance:

1. **Proper nouns over verbs.** `raspberry pi 5 bookworm kernel version`
   beats `how do I check kernel version on raspberry pi`.
2. **Three to six keywords.** Longer prose queries hurt recall on DDG.
3. **`site:` is your friend** when you know a good source:
   `site:docs.python.org pathlib resolve`.
4. **Quote exact error strings**, but drop file paths and line numbers:
   `"sqlite3.OperationalError: no such column"`.
5. **Add a year** if recency matters: `python 3.14 release notes`.

## Reading results

The tool returns a ranked list of `{title, url, snippet}`. Don't stop at
the snippets — they're abstracts and often misleading. `web_fetch` the
top 1–3 URLs whose title *and* snippet actually match before committing
to an answer. Cite the URL in your reply so the user can verify.

## Common failure modes

- **Zero results** — query is too specific or too rephrased. Simplify:
  drop adjectives, try synonyms, or add a `site:` hint for a known good
  source.
- **Top result is a low-quality aggregator** — skip past it. Official
  docs, StackOverflow, GitHub issues, and reputable blogs are almost
  always better than scraper sites.
- **Sources contradict each other** — prefer the more recent and more
  authoritative one, and *say so* in your reply: "per X's 2025 docs"
  beats "per the web".
- **Live lookups** (stock prices, weather, live scores) — DDG's index
  lag makes this a bad fit. Tell the user the limitation and suggest a
  different approach.
