"""Retrieval-efficacy harness: does each retrieval stage earn its tokens?

Scores nano-hermes's retrieval arms against gold evidence ids from a
script-first corpus (see ``corpus.py``). This layer needs **no LLM and no
judge** — the corpus knows which chunk states each fact, so recall/MRR are
computed exactly. That removes judge bias from the component question we most
want answered, and lets the whole thing run in CI.

Arms are deliberately split so each stage is attributable:

    fts_only   — lexical channel alone (BM25 via FTS5)
    vec_only   — dense channel alone (sqlite-vec ANN)
    rrf        — both channels fused, NO diversity rerank
    rrf_mmr    — the shipped path: fusion + MMR
    full_ctx   — every chunk (recall ceiling, token ceiling)

``rrf`` vs ``rrf_mmr`` is the point of the split: MMR is a *diversity*
reranker, and no published evaluation of it in an agent-memory setting could be
found, so it is the component most likely to be free-riding. Likewise
``fts_only`` vs ``rrf`` tests fusion against the strong lexical baseline rather
than against a weak cosine-only one — in this domain (exact identifiers: error
codes, paths, PR numbers) lexical matching is unusually hard to beat.

WHAT THIS DOES NOT MEASURE: offline ranking quality is not a reliable proxy for
whether retrieved evidence actually helps the model answer (arXiv:2601.17532).
Surfacing the gold chunk is necessary, not sufficient. An answer+judge layer is
the separate, LLM-dependent half.

Embeddings: ``--embedder fake`` is deterministic and offline, for exercising
the plumbing in CI. Only ``--embedder real`` produces numbers worth citing —
with a fake embedder the dense arm is meaningless by construction.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from evals.corpus import Corpus, generate_corpus

ARMS = ("fts_only", "vec_only", "rrf", "rrf_mmr", "full_ctx")


@dataclass
class ArmResult:
    arm: str
    n_questions: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    mean_injected_tokens: float
    median_latency_ms: float
    k: int


def _tokens(text: str) -> int:
    """Approximate token count. tiktoken is already a dependency."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def seed_db(db, corpus: Corpus) -> dict[tuple[int, int], int]:
    """Insert the corpus; return {(session_idx, turn_index): chunk_id}."""
    now = time.time()
    ids: dict[tuple[int, int], int] = {}
    sess_ids: dict[int, int] = {}
    for s in sorted({c.session_idx for c in corpus.chunks}):
        cur = db.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
            (f"eval-s{s}", now + s),
        )
        sess_ids[s] = int(cur.lastrowid)
    for c in corpus.chunks:
        cur = db.execute(
            "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (sess_ids[c.session_idx], c.turn_index, c.role, c.content,
             now + c.session_idx * 100 + c.turn_index),
        )
        ids[(c.session_idx, c.turn_index)] = int(cur.lastrowid)
    db.commit()
    return ids


def _retrieve(arm: str, db, query: str, qvec: np.ndarray, cfg, k: int) -> list[int]:
    """Return ranked chunk ids for *arm*."""
    from nano_hermes.session.search import (
        _fts_rows,
        hybrid_search,
        reciprocal_rank_fusion,
    )

    if arm == "full_ctx":
        return [r[0] for r in db.execute("SELECT id FROM chunks ORDER BY id").fetchall()]

    if arm == "fts_only":
        rows = _fts_rows(
            db,
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            query, k,
        )
        return [r[0] for r in rows]

    # vec_only is a direct top-k baseline, so it asks for exactly k. The fusion
    # arms must instead use cfg.vec_k — the width hybrid_search uses — or `rrf`
    # and `rrf_mmr` would differ in candidate-pool width AS WELL AS the MMR
    # rerank, confounding the single comparison this split exists to make.
    pool = k if arm == "vec_only" else cfg.vec_k
    vec_rows = db.execute(
        "SELECT chunk_id FROM chunks_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (qvec.astype(np.float32).tobytes(), pool),
    ).fetchall()
    vec_ids = [r[0] for r in vec_rows]
    if arm == "vec_only":
        return vec_ids

    if arm == "rrf":
        fts_rows = _fts_rows(
            db,
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            query, cfg.fts_k,
        )
        fused = reciprocal_rank_fusion([r[0] for r in fts_rows], vec_ids, cfg.rrf_k)
        return [cid for cid, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)][:k]

    if arm == "rrf_mmr":
        return [h.chunk_id for h in hybrid_search(db, query, qvec, cfg)]

    raise ValueError(f"unknown arm {arm!r}")


def score(ranked: list[int], gold: set[int], k: int) -> tuple[float, float, float]:
    """(recall@k, reciprocal rank, nDCG@k) for one question."""
    if not gold:
        return (math.nan, math.nan, math.nan)   # abstention: no evidence to find
    top = ranked[:k]
    hit = len(set(top) & gold)
    recall = hit / len(gold)
    rr = 0.0
    for i, cid in enumerate(top, start=1):
        if cid in gold:
            rr = 1.0 / i
            break
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(top, start=1) if cid in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return (recall, rr, dcg / ideal if ideal else 0.0)


async def _embed(texts: list[str], mode: str, dims: int) -> list[np.ndarray]:
    if mode == "fake":
        out = []
        for t in texts:
            # Stable across processes: builtin hash() is salted per-process by
            # PYTHONHASHSEED, which would silently break the documented
            # determinism of --embedder fake.
            digest = hashlib.sha256(t.encode("utf-8")).digest()[:4]
            rng = np.random.default_rng(int.from_bytes(digest, "big"))
            v = rng.standard_normal(dims).astype(np.float32)
            out.append(v / (np.linalg.norm(v) or 1.0))
        return out
    # Real path: uses the same provider chain the agent uses, so the dense arm
    # is measured on the vectors nano-hermes would actually store.
    from nano_hermes.config import NanoHermesConfig  # noqa: PLC0415
    from nano_hermes.embedding.chain import EmbeddingChain  # noqa: PLC0415

    cfg = NanoHermesConfig()
    async with EmbeddingChain(cfg.embedding) as chain:
        return list(await chain.embed(texts))


async def run(
    *, seed: int, k: int, embedder: str, out: Path | None, corpus_kwargs: dict
) -> list[ArmResult]:
    import tempfile

    from nano_hermes.config import RetrievalConfig
    from nano_hermes.session.db import open_db

    corpus = generate_corpus(seed=seed, **corpus_kwargs)
    # Read the dimension from the same config the real embedder builds, so the
    # dense arms are measured at the width nano-hermes actually stores rather
    # than a hard-coded default.
    from nano_hermes.config import NanoHermesConfig  # noqa: PLC0415

    dims = NanoHermesConfig().embedding.target_dims
    tmp = Path(tempfile.mkdtemp())
    db = open_db(str(tmp / "eval.db"), dims)

    texts = [c.content for c in corpus.chunks]
    chunk_vecs = await _embed(texts, embedder, dims)
    ids = seed_db(db, corpus)
    vec_by_id = {ids[(c.session_idx, c.turn_index)]: v
                 for c, v in zip(corpus.chunks, chunk_vecs)}
    for cid, v in vec_by_id.items():
        db.execute("INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                   (cid, v.astype(np.float32).tobytes()))
    db.commit()

    qvecs = await _embed([q.text for q in corpus.questions], embedder, dims)
    cfg = RetrievalConfig(final_k=k)
    content_by_id = {cid: c.content for c, cid in
                     ((c, ids[(c.session_idx, c.turn_index)]) for c in corpus.chunks)}

    results: list[ArmResult] = []
    for arm in ARMS:
        recs, rrs, ndcgs, toks, lats = [], [], [], [], []
        for q, qv in zip(corpus.questions, qvecs):
            gold = {ids[key] for key in q.gold_chunk_keys}
            t0 = time.perf_counter()
            ranked = _retrieve(arm, db, q.text, qv, cfg, k)
            lats.append((time.perf_counter() - t0) * 1000)
            # full_ctx puts the entire history in the prompt; truncating it to k
            # would measure "the first k chunks by id", not the arm. Score it
            # over everything it actually injects — that is the whole point of
            # carrying it as the ceiling baseline.
            eff_k = len(ranked) if arm == "full_ctx" else k
            r, rr, nd = score(ranked, gold, eff_k)
            if not math.isnan(r):
                recs.append(r)
                rrs.append(rr)
                ndcgs.append(nd)
            toks.append(sum(_tokens(content_by_id[c])
                            for c in ranked[:eff_k] if c in content_by_id))
        results.append(ArmResult(
            arm=arm, n_questions=len(recs),
            recall_at_k=round(float(np.mean(recs)), 4),
            mrr=round(float(np.mean(rrs)), 4),
            ndcg_at_k=round(float(np.mean(ndcgs)), 4),
            mean_injected_tokens=round(float(np.mean(toks)), 1),
            median_latency_ms=round(float(np.median(lats)), 3),
            k=k,
        ))

    db.close()
    shutil.rmtree(tmp, ignore_errors=True)

    if out:
        out.write_text(json.dumps(
            {"seed": seed, "k": k, "embedder": embedder,
             "corpus": {"facts": len(corpus.facts), "chunks": len(corpus.chunks),
                        "sessions": corpus.n_sessions, "questions": len(corpus.questions)},
             "arms": [asdict(x) for x in results]}, indent=2))
    return results


def format_table(results: list[ArmResult], embedder: str) -> str:
    head = f"{'arm':10s} {'recall@k':>9s} {'MRR':>7s} {'nDCG@k':>7s} {'tokens':>9s} {'ms':>7s}"
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{r.arm:10s} {r.recall_at_k:9.3f} {r.mrr:7.3f} {r.ndcg_at_k:7.3f} "
            f"{r.mean_injected_tokens:9.1f} {r.median_latency_ms:7.2f}"
        )
    if embedder == "fake":
        lines.append("")
        lines.append("NOTE: --embedder fake -> vec_only, rrf and rrf_mmr are all")
        lines.append("meaningless by construction (random vectors; MMR diversifies")
        lines.append("over noise). Only fts_only is interpretable here. Plumbing")
        lines.append("check only -- do not cite these numbers. Use --embedder real.")
    lines.append("")
    lines.append("Single tenure: ranking is NOT safe to generalise (arXiv:2607.21962).")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--embedder", choices=("fake", "real"), default="fake")
    ap.add_argument("--facts", type=int, default=24)
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res = asyncio.run(run(
        seed=a.seed, k=a.k, embedder=a.embedder, out=a.out,
        corpus_kwargs={"n_facts": a.facts, "sessions": a.sessions},
    ))
    print(format_table(res, a.embedder))


if __name__ == "__main__":
    main()
