"""LongMemEval_S session-retrieval eval: nano-hermes's retrieval on a public set.

LongMemEval (arXiv:2410.10813, MIT) gives each question a haystack of ~50 chat
sessions and the ids of the sessions holding the evidence
(``answer_session_ids``). That makes session-level retrieval scorable with
**no LLM and no judge**: ingest the haystack the way nano-hermes stores it (one
chunk per message, embedded with the archiver's task preamble), run each arm of
``evals.harness``, map the ranked chunks to their sessions, and score.

What the number is and is not:

* It is retrieval — whether the evidence session reaches the prompt. It is NOT
  the QA accuracy memory leaderboards report (Mem0, MemOS, ...), which add a
  frontier answerer and a judge on top.
* The paper reports session-level retrieval only on LongMemEval_M (Table 9:
  BM25 R@5 0.634 / R@10 0.710, Contriever 0.723 / 0.823, Stella-1.5B 0.720 /
  0.794, raw-session keys). S has ~10x fewer distractor sessions, so S numbers
  run higher and are not directly comparable; ``fts_only`` here is the
  in-house BM25 reference.
* The 30 abstention questions (``*_abs``) are skipped, as in the paper.

Data (not downloaded for you; ~265 MB)::

    wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

    uv run python -m evals.longmemeval --data longmemeval_s_cleaned.json --embedder fake --limit 5
    uv run python -m evals.longmemeval --data longmemeval_s_cleaned.json --embedder real --limit 50

``--embedder real`` costs roughly 115k tokens per question (~6M for 50).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evals.corpus import Chunk
from evals.harness import _embed, _retrieve, seed_db

ARMS = ("fts_only", "vec_only", "rrf", "rrf_mmr")
POOL = 50        # chunks retrieved per arm before collapsing to sessions
_BATCH = 64      # texts per embedding request


@dataclass
class _Haystack:
    chunks: list[Chunk]


def load(path: Path, limit: int | None) -> list[dict]:
    items = [q for q in json.loads(path.read_text()) if not q["question_id"].endswith("_abs")]
    return items[:limit] if limit else items


def session_scores(ranked: list[str], gold: set[str]) -> dict[str, float]:
    """recall_any@k / recall_all@k (k=5, 10) and nDCG@10 over ranked session ids."""
    out: dict[str, float] = {}
    for k in (5, 10):
        top = set(ranked[:k])
        out[f"any@{k}"] = float(bool(top & gold))
        out[f"all@{k}"] = float(gold <= top)
    dcg = sum(1 / math.log2(i + 2) for i, s in enumerate(ranked[:10]) if s in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    out["ndcg@10"] = dcg / ideal if ideal else 0.0
    return out


async def _embed_batched(texts: list[str], embedder: str, dims: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i in range(0, len(texts), _BATCH):
        out.extend(await _embed(texts[i:i + _BATCH], embedder, dims))
    return out


async def score_question(item: dict, embedder: str, dims: int) -> dict[str, dict[str, float]]:
    from nano_hermes.config import RetrievalConfig
    from nano_hermes.embedding.contextual import add_context_preamble
    from nano_hermes.session.db import open_db

    chunks: list[Chunk] = []
    embed_texts: list[str] = []
    for s_idx, session in enumerate(item["haystack_sessions"]):
        # The archiver prefixes each chunk's embedding text with the session's
        # first user message (archiver.py `_session_tasks`, `add_context_preamble`).
        task = next((t["content"][:500] for t in session if t["role"] == "user"), None)
        for turn, t in enumerate(session):
            if not t.get("content"):
                continue
            chunks.append(Chunk(s_idx, turn, t["role"], t["content"]))
            embed_texts.append(add_context_preamble(t["content"], task))

    tmp = Path(tempfile.mkdtemp())
    try:
        db = open_db(str(tmp / "lme.db"), dims)
        ids = seed_db(db, _Haystack(chunks))
        vecs = await _embed_batched(embed_texts, embedder, dims)
        for c, v in zip(chunks, vecs):
            db.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                (ids[(c.session_idx, c.turn_index)], v.astype(np.float32).tobytes()),
            )
        db.commit()
        ext_by_chunk = {
            ids[(c.session_idx, c.turn_index)]: item["haystack_session_ids"][c.session_idx]
            for c in chunks
        }
        [qvec] = await _embed([item["question"]], embedder, dims)
        cfg = RetrievalConfig(final_k=POOL, fts_k=POOL, vec_k=POOL)
        gold = set(item["answer_session_ids"])
        out = {}
        for arm in ARMS:
            ranked = list(dict.fromkeys(  # chunks → sessions, first hit wins
                ext_by_chunk[c] for c in _retrieve(arm, db, item["question"], qvec, cfg, POOL)
            ))
            out[arm] = session_scores(ranked, gold)
        db.close()
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def run(data: Path, limit: int | None, embedder: str) -> dict:
    from nano_hermes.config import NanoHermesConfig

    dims = NanoHermesConfig().embedding.target_dims
    items = load(data, limit)
    # per arm: overall metric lists, and per question_type metric lists
    overall: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_type: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item in items:
        for arm, m in (await score_question(item, embedder, dims)).items():
            for name, v in m.items():
                overall[arm][name].append(v)
            by_type[(arm, item["question_type"])].append(m["any@5"])
    return {
        "n": len(items),
        "embedder": embedder,
        "arms": {a: {n: round(float(np.mean(v)), 4) for n, v in ms.items()} for a, ms in overall.items()},
        "any@5_by_type": {f"{a}/{t}": round(float(np.mean(v)), 4) for (a, t), v in sorted(by_type.items())},
    }


def format_table(res: dict) -> str:
    cols = ("any@5", "all@5", "any@10", "all@10", "ndcg@10")
    head = f"{'arm':10s}" + "".join(f"{c:>9s}" for c in cols)
    lines = [f"LongMemEval_S session retrieval, n={res['n']}, embedder={res['embedder']}", head, "-" * len(head)]
    for arm, m in res["arms"].items():
        lines.append(f"{arm:10s}" + "".join(f"{m[c]:9.3f}" for c in cols))
    if res["embedder"] == "fake":
        lines += ["", "NOTE: --embedder fake: only fts_only is meaningful. Do not cite."]
    lines += ["", "Retrieval only, not QA accuracy; the paper's retrieval table is on M, not S."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=50, help="questions (0 = all 470)")
    ap.add_argument("--embedder", choices=("fake", "real"), default="fake")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res = asyncio.run(run(a.data, a.limit or None, a.embedder))
    if a.out:
        a.out.write_text(json.dumps(res, indent=2))
    print(format_table(res))


if __name__ == "__main__":
    main()
