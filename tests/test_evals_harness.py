"""Tests for the offline retrieval-efficacy harness (evals/).

The harness is a dev tool, not shipped runtime, but it makes claims about
nano-hermes's retrieval quality — so the corpus invariants that make those
claims meaningful are worth pinning. A corpus with ambiguous gold or with the
answer restated in a distractor would produce confident, wrong conclusions.
"""
from __future__ import annotations

import asyncio

import pytest

from evals.corpus import generate_corpus
from evals.harness import ARMS, run, score


class TestCorpusInvariants:
    def test_subject_attr_pairs_are_unique(self):
        # Two live facts for the same (subject, attr) would make "the current X
        # for Y" ambiguous and silently mis-score correct answers.
        c = generate_corpus()
        pairs = [(f.subject, f.attr) for f in c.facts if f.version == 0]
        assert len(pairs) == len(set(pairs))

    def test_too_many_facts_is_refused_not_silently_duplicated(self):
        with pytest.raises(ValueError, match="unique"):
            generate_corpus(n_facts=10_000)

    def test_knowledge_update_gold_is_the_new_value(self):
        c = generate_corpus()
        updates = {f.fact_id: f for f in c.facts if f.version == 1}
        originals = {f.fact_id: f for f in c.facts if f.version == 0}
        qs = [q for q in c.questions if q.kind == "knowledge_update"]
        assert qs, "corpus produced no knowledge-update questions"
        for q in qs:
            assert q.gold == updates[q.gold_fact_id].value
            assert q.gold != originals[q.gold_fact_id].value

    def test_update_is_rendered_after_the_original(self):
        # "Current" must mean "later in the timeline", or the update question
        # is unanswerable even with perfect retrieval.
        c = generate_corpus()
        order: dict[str, list[int]] = {}
        for i, ch in enumerate(c.chunks):
            if ch.fact_id:
                order.setdefault(ch.fact_id, []).append(i)
        for fid, positions in order.items():
            assert positions == sorted(positions)

    def test_abstention_questions_have_no_evidence(self):
        c = generate_corpus()
        abs_qs = [q for q in c.questions if q.kind == "abstention"]
        assert abs_qs
        for q in abs_qs:
            assert q.gold_chunk_keys == []
            assert q.gold == "NOT IN CONTEXT"

    def test_gold_value_appears_verbatim_in_its_evidence_chunk(self):
        # Gold is correct by construction only if the planted value really is
        # in the chunk the answer key points at.
        c = generate_corpus()
        by_key = {(ch.session_idx, ch.turn_index): ch for ch in c.chunks}
        checked = 0
        for q in c.questions:
            for key in q.gold_chunk_keys:
                assert q.gold in by_key[key].content
                checked += 1
        assert checked > 0

    def test_deterministic_for_a_seed(self):
        a, b = generate_corpus(seed=3), generate_corpus(seed=3)
        assert [f.value for f in a.facts] == [f.value for f in b.facts]
        assert [q.gold for q in a.questions] == [q.gold for q in b.questions]


class TestScoring:
    def test_perfect_ranking(self):
        r, rr, nd = score([1, 2, 3], {1}, 3)
        assert (r, rr) == (1.0, 1.0)
        assert nd == pytest.approx(1.0)

    def test_miss_scores_zero(self):
        r, rr, nd = score([4, 5, 6], {1}, 3)
        assert (r, rr, nd) == (0.0, 0.0, 0.0)

    def test_rank_position_matters(self):
        _, rr_first, _ = score([1, 9, 9], {1}, 3)
        _, rr_third, _ = score([9, 9, 1], {1}, 3)
        assert rr_first > rr_third

    def test_abstention_is_excluded_not_scored_zero(self):
        # No evidence exists, so counting it as a miss would understate every
        # arm equally and dilute the comparison.
        import math
        r, rr, nd = score([1, 2], set(), 2)
        assert math.isnan(r) and math.isnan(rr) and math.isnan(nd)


@pytest.fixture(scope="module")
def results():
    """One harness run shared by the end-to-end assertions."""
    return asyncio.run(run(
        seed=7, k=8, embedder="fake", out=None,
        corpus_kwargs={"n_facts": 12, "sessions": 6},
    ))


class TestHarnessEndToEnd:

    def test_every_arm_reports(self, results):
        assert [r.arm for r in results] == list(ARMS)
        assert all(r.n_questions > 0 for r in results)

    def test_lexical_arm_finds_planted_facts(self, results):
        # The only arm that is meaningful under the fake embedder: planted
        # values are rare tokens, so FTS should surface them.
        fts = next(r for r in results if r.arm == "fts_only")
        assert fts.recall_at_k > 0.8

    def test_full_context_is_the_recall_and_token_ceiling(self, results):
        # Regression guard: full_ctx was once truncated to k, which measured
        # "the first k chunks by id" and reported recall 0.125.
        full = next(r for r in results if r.arm == "full_ctx")
        assert full.recall_at_k == 1.0
        assert full.mean_injected_tokens == max(r.mean_injected_tokens for r in results)

    def test_retrieval_arms_are_cheaper_than_full_context(self, results):
        full = next(r for r in results if r.arm == "full_ctx")
        for r in results:
            if r.arm != "full_ctx":
                assert r.mean_injected_tokens < full.mean_injected_tokens


class TestFakeEmbedderDeterminism:
    """--embedder fake is documented as deterministic. It must survive a new
    process: builtin hash() is salted by PYTHONHASHSEED, which silently broke
    this once already."""

    def test_stable_across_processes_and_hash_seeds(self):
        import os
        import subprocess
        import sys

        code = (
            "import asyncio,json;"
            "from evals.harness import _embed;"
            "v=asyncio.run(_embed(['alpha','beta'],'fake',8));"
            "print(json.dumps([[round(float(x),6) for x in a] for a in v]))"
        )
        outs = []
        for seed in ("1", "424242"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            r = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, cwd=os.getcwd(), env=env, check=True,
            )
            outs.append(r.stdout.strip())
        assert outs[0] == outs[1], "fake embedder changed with PYTHONHASHSEED"


class TestFusionArmsAreComparable:
    """`rrf` vs `rrf_mmr` is the comparison the harness exists for, so the two
    arms must differ ONLY by the MMR rerank — not by candidate-pool width."""

    def test_fusion_arms_request_the_same_vector_pool(self, tmp_path):
        import numpy as np

        from nano_hermes.config import RetrievalConfig
        from nano_hermes.session.db import open_db
        from evals.harness import _retrieve

        db = open_db(str(tmp_path / "p.db"), 8)
        db.execute("INSERT INTO sessions (session_key, started_at) VALUES ('s', 0)")
        db.execute(
            "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
            "VALUES (1, 0, 'user', 'rollback billing-api E1234', 0)"
        )
        db.commit()
        qv = np.zeros(8, dtype=np.float32)
        qv[0] = 1.0
        db.execute("INSERT INTO chunks_vec (chunk_id, embedding) VALUES (1, ?)",
                   (qv.tobytes(),))
        db.commit()

        cfg = RetrievalConfig(final_k=4)
        assert cfg.vec_k != 4, "test needs vec_k to differ from k to be meaningful"

        widths: list[int] = []

        class _Spy:
            """sqlite3.Connection.execute is read-only, so wrap rather than patch."""

            def __init__(self, conn):
                self._conn = conn

            def execute(self, sql, params=()):
                if "chunks_vec" in sql and "MATCH" in sql:
                    widths.append(params[1])
                return self._conn.execute(sql, params)

            def __getattr__(self, name):
                return getattr(self._conn, name)

        spy = _Spy(db)
        _retrieve("rrf", spy, "rollback", qv, cfg, 4)
        _retrieve("rrf_mmr", spy, "rollback", qv, cfg, 4)

        assert widths, "no vector query observed"
        assert len(set(widths)) == 1, (
            f"fusion arms used different vector pool widths {widths} — the "
            "rrf/rrf_mmr comparison would confound MMR with pool size"
        )
        assert widths[0] == cfg.vec_k
