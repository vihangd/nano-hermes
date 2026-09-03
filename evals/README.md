# Retrieval-efficacy harness

Answers one question: **does each retrieval stage earn the tokens it spends?**

nano-hermes ships ~20 memory/evolution features. Its 1,294 tests prove they are
*correct*; none of them showed any of it *helps*. This harness is the first
evidence layer. It is a development tool — it is not imported by the plugin and
never runs on the Pi.

```bash
uv run python -m evals.harness --embedder fake            # plumbing check, no network
uv run python -m evals.harness --embedder real --k 8      # real numbers (needs an embedding key)
uv run python -m evals.harness --embedder real --out results.json
```

## What it measures

Retrieval quality against **gold evidence ids**, with no LLM and no judge:
`recall@k`, `MRR`, `nDCG@k`, mean **injected tokens** and median latency per
question. The accuracy-vs-tokens pair is the answer; a recall number on its own
is not.

Note that every question has exactly one gold chunk, so ideal-DCG is always 1
and **nDCG collapses to a transform of the reciprocal rank** — read `MRR` and
`nDCG@k` as one signal, not two corroborating ones.

Arms are split so each stage is attributable:

| arm | what it isolates |
|---|---|
| `fts_only` | lexical channel alone (BM25 via FTS5) |
| `vec_only` | dense channel alone (sqlite-vec ANN) |
| `rrf` | both channels fused, **no** diversity rerank |
| `rrf_mmr` | the shipped path: fusion + MMR |
| `full_ctx` | every chunk — the recall ceiling and the token ceiling |

Two comparisons carry the weight:

* **`rrf` vs `rrf_mmr`** — MMR is a *diversity* reranker and no published
  evaluation of it in an agent-memory setting could be found. It is the
  component most likely to be free-riding, so it gets its own arm.
* **`fts_only` vs `rrf`** — fusion is tested against the *strong* lexical
  baseline, not the weak cosine-only baseline commonly used. In this domain
  (error codes, paths, PR numbers) exact-identifier matching is hard to beat,
  and at least one adjacent result reports BM25 as the single most important
  channel.

`full_ctx` is the arm that kills most memory systems: if dumping everything in
scores the same, the memory layer is buying nothing but latency.

## Why the corpus is generated fact-first

Facts are created **before** the transcripts that mention them
([Ground Truth First, arXiv:2607.21962](https://arxiv.org/abs/2607.21962)):

* Gold is right by construction — no LLM reads a transcript and guesses a key.
* Each fact carries a version, so a superseded value is a *different version*
  rather than a contradiction. A "current value" question has exactly one right
  answer, and an earlier mention cannot masquerade as the answer.
* Every rendered chunk records the fact it states, so gold **evidence ids** are
  free — which is what removes the LLM from the scoring loop entirely.

The domain is agentic-coding (services, deploys, error codes, PR numbers,
paths) rather than the chat/email life-scripts the published instruments use,
because that is what nano-hermes actually retrieves over. Synthetic also means
no real session data leaves the machine.

Question kinds: `extraction`, `knowledge_update` (answer is the *newer* value),
`abstention` (the subject does not exist; excluded from recall rather than
scored as a miss).

## Limitations — read before citing any number

1. **Ranking quality is not answer quality.** Surfacing the gold chunk is
   necessary, not sufficient; offline ranking metrics are a known-unreliable
   proxy for whether evidence helps the generator
   ([arXiv:2601.17532](https://arxiv.org/pdf/2601.17532)). An answer+judge layer
   is the separate, LLM-dependent half and is not built yet.
2. **`--embedder fake` produces meaningless dense arms.** Random vectors mean
   `vec_only`, `rrf` and `rrf_mmr` are noise, and MMR diversifies over noise.
   Only `fts_only` is interpretable. It exists to exercise plumbing in CI.
3. **Single tenure.** Memory-architecture rankings are known to *invert* with
   history length — the "tenure crossover" — so a one-timepoint result can rank
   arms backwards. Tenure is a corpus parameter here, deliberately, so adding
   points is a config change rather than a rewrite. **Do not generalise a
   ranking from a single tenure.**
4. **Small bank.** At ~30 questions only large effects are resolvable. If two
   arms differ by a few points, the honest reading is "not resolvable at this
   bank size", not "arm A wins".
5. **`--embedder fake` is deterministic, but only the metrics are.** Latency
   columns vary run to run, as timings do.
6. **No parametric-knowledge filter yet.** The published control is to drop any
   question a no-memory model answers correctly. Planted values are random
   identifiers, so leakage is unlikely by construction — but it is not measured.

## Roadmap

* Tenure curve — score the same bank at ≥2 history lengths and report the
  crossover.
* Answer + judge layer — pointwise 2/1/0 rubric, fixed answerer, version-pinned
  judge, ≥3 stochastic replicates, cross-family re-judge of the headline.
* Parametric-leakage filter over the bank.
* Distilled-artifact arms — reflections / principles / cheatsheet lessons
  ablated independently, not just retrieval.
* Paired bootstrap CIs and McNemar for the `fts_only` vs `rrf` head-to-head.
