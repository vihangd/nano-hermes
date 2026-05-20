"""Regression guard: vec0 tables must use cosine distance metric."""
import numpy as np
import nano_hermes
from conftest import _make_loop


class TestVec0CosineMetric:
    def test_chunks_vec_uses_cosine_distance(self, tmp_path):
        """Two identical unit vectors should have distance ≈ 0 (cosine distance = 1 - similarity)."""
        import time as _time

        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        db = hook.db
        dims = hook.config.embedding.target_dims

        v = np.zeros(dims, dtype=np.float32)
        v[0] = 1.0  # unit vector

        cur = db.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
            ("vec0-cosine-test", _time.time()),
        )
        sid = cur.lastrowid
        cur2 = db.execute(
            "INSERT INTO chunks (session_id, turn_index, role, content, created_at) VALUES (?, 0, 'user', 'x', ?)",
            (sid, _time.time()),
        )
        chunk_id = cur2.lastrowid
        db.execute("INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)", (chunk_id, v.tobytes()))
        db.commit()

        rows = db.execute(
            "SELECT chunk_id, distance FROM chunks_vec WHERE embedding MATCH ? AND k = 1",
            (v.tobytes(),),
        ).fetchall()
        assert rows, "no results from chunks_vec"
        dist = rows[0][1]
        # cosine distance between identical vectors = 0.0 (within float tolerance)
        assert dist < 0.01, (
            f"chunks_vec returned distance {dist:.4f} for identical vectors — "
            "expected ~0.0 (cosine); got L2-like value suggesting distance_metric=cosine is missing"
        )

    def test_orthogonal_vectors_have_distance_near_1(self, tmp_path):
        """Orthogonal unit vectors have cosine_sim=0 → cosine_distance=1."""
        import time as _time

        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        db = hook.db
        dims = hook.config.embedding.target_dims

        v0 = np.zeros(dims, dtype=np.float32)
        v0[0] = 1.0
        v1 = np.zeros(dims, dtype=np.float32)
        v1[1] = 1.0

        cur = db.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
            ("vec0-ortho-test", _time.time()),
        )
        sid = cur.lastrowid
        cur2 = db.execute(
            "INSERT INTO chunks (session_id, turn_index, role, content, created_at) VALUES (?, 0, 'user', 'a', ?)",
            (sid, _time.time()),
        )
        cid1 = cur2.lastrowid
        cur3 = db.execute(
            "INSERT INTO chunks (session_id, turn_index, role, content, created_at) VALUES (?, 1, 'user', 'b', ?)",
            (sid, _time.time()),
        )
        cid2 = cur3.lastrowid
        db.execute("INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)", (cid1, v0.tobytes()))
        db.execute("INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)", (cid2, v1.tobytes()))
        db.commit()

        rows = db.execute(
            "SELECT chunk_id, distance FROM chunks_vec WHERE embedding MATCH ? AND k = 2",
            (v0.tobytes(),),
        ).fetchall()
        assert len(rows) == 2
        dists = {r[0]: r[1] for r in rows}
        assert dists[cid1] < 0.01, "self-distance should be ~0"
        assert dists[cid2] > 0.9, (
            f"orthogonal vector distance was {dists[cid2]:.4f} — expected ~1.0 (cosine); "
            "L2 distance for orthogonal unit vectors is sqrt(2)≈1.414, which > 1.0 also fails this bound"
        )
