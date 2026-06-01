"""Session archive schema, workspace-relative.

The whole archive lives in one SQLite file at
``<workspace>/nano_hermes/state.db``. Core concerns:

    sessions       — session metadata (rolling retention lives here)
    chunks         — turn-level content rows
    chunks_fts     — FTS5 mirror for keyword search (external content)
    chunks_vec     — sqlite-vec vec0 for embedding search

Phase 2 tables (declared up-front so the schema stays stable):

    skill_stats    — mutable skill state (use_count, status, provenance…)
    trajectories   — replay buffer for similar-task retrieval
    reflections    — session-scoped Reflexion entries

External-content FTS5 keeps ``chunks`` canonical and mirrors insertions /
deletions via triggers. Rolling retention (default 45 days) is applied by
``purge_older_than``, which should run from the nanobot dream cycle — not
per turn.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import sqlite_vec

from ..paths import state_db

log = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key  TEXT NOT NULL,
    started_at   REAL NOT NULL,
    ended_at     REAL,
    summary      TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_key      ON sessions(session_key);
CREATE INDEX IF NOT EXISTS idx_sessions_ended_at ON sessions(ended_at);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_index  INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='id',
    tokenize='porter'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

-- Phase 2 tables ---------------------------------------------------------

CREATE TABLE IF NOT EXISTS skill_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT, -- stable rowid for skill_vec
    name            TEXT NOT NULL UNIQUE,
    status          TEXT NOT NULL DEFAULT 'active',   -- draft | active | deprecated
    use_count       INTEGER NOT NULL DEFAULT 0,
    success_count   INTEGER NOT NULL DEFAULT 0,
    last_used_at    REAL,
    provenance      TEXT,                              -- JSON list of session ids
    content_hash    TEXT,                              -- sha1 of (name + description)
    indexed_at      REAL                               -- last time we embedded this skill
);

-- Small key/value store for one-off metadata that doesn't justify its
-- own table (curator cooldown, etc). Values are stored as TEXT — callers
-- coerce as needed.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trajectories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    task            TEXT NOT NULL,
    skills_used     TEXT,                              -- JSON list
    outcome         TEXT NOT NULL,                     -- ok | fail | partial
    reflection      TEXT,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trajectories_created ON trajectories(created_at);

CREATE TABLE IF NOT EXISTS reflections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    content         TEXT NOT NULL,
    created_at      REAL NOT NULL,
    view_count      INTEGER NOT NULL DEFAULT 0,
    cite_count      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reflections_session ON reflections(session_id);

-- Phase 3 tables ---------------------------------------------------------

CREATE TABLE IF NOT EXISTS skill_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name  TEXT NOT NULL,
    body        TEXT NOT NULL,           -- snapshot of SKILL.md at version time
    reason      TEXT,                    -- e.g. "auto-rewrite by SkillRewriter"
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_skill_versions_name ON skill_versions(skill_name, created_at);

CREATE TABLE IF NOT EXISTS skill_compositions (
    skill_a     TEXT NOT NULL,
    skill_b     TEXT NOT NULL,
    count       INTEGER NOT NULL DEFAULT 1,
    last_used   REAL NOT NULL,
    UNIQUE(skill_a, skill_b)
);
CREATE INDEX IF NOT EXISTS idx_skill_compositions_a ON skill_compositions(skill_a, count DESC);

-- Phase 5: Structured principles (EvolveR pattern).
-- Generalised agent-authored rules with condition/action/expected_outcome.
-- Principles are discovered by FTS5 match of the current task against condition text.
CREATE TABLE IF NOT EXISTS principles (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    condition        TEXT NOT NULL,     -- "when X happens"
    action           TEXT NOT NULL,     -- "do Y"
    expected_outcome TEXT,              -- "so that Z"
    confidence       REAL NOT NULL DEFAULT 0.5,
    created_at       REAL NOT NULL,
    use_count        INTEGER NOT NULL DEFAULT 0,
    success_count    INTEGER NOT NULL DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS principles_fts USING fts5(
    condition,
    action,
    content_id UNINDEXED,
    tokenize='porter'
);

CREATE TRIGGER IF NOT EXISTS principles_ad AFTER DELETE ON principles BEGIN
    INSERT INTO principles_fts(principles_fts, rowid, condition, action, content_id)
    VALUES ('delete', old.id, old.condition, old.action, old.id);
END;

-- Phase 5: Associative reflection co-activation graph (HeLa-Mem-lite).
-- When two reflections are injected together in the same session iteration,
-- their co-activation is recorded. Highly co-activated pairs signal thematic
-- associations useful for future retrieval ranking or pruning.
CREATE TABLE IF NOT EXISTS reflection_coactivations (
    reflection_a_id  INTEGER NOT NULL,
    reflection_b_id  INTEGER NOT NULL,
    coactivation_count INTEGER NOT NULL DEFAULT 1,
    last_at          REAL NOT NULL,
    PRIMARY KEY (reflection_a_id, reflection_b_id)
);
CREATE INDEX IF NOT EXISTS idx_reflection_coact_a ON reflection_coactivations(reflection_a_id, coactivation_count DESC);

-- Phase 4.3: Episodic→semantic distillation output.
-- LLM-distilled facts from hub clusters; source_chunk_ids is a JSON array
-- of chunk_id ints for poisoning traceability.
-- A-MEM (Phase 6): keywords/tags/context are LLM-authored note attributes;
-- importance is 1-10 (Generative-Agents-style).
CREATE TABLE IF NOT EXISTS semantic_facts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    content          TEXT NOT NULL,
    source_chunk_ids TEXT NOT NULL,
    created_at       REAL NOT NULL,
    keywords         TEXT NOT NULL DEFAULT '[]',  -- JSON array of strings
    tags             TEXT NOT NULL DEFAULT '[]',  -- JSON array of strings
    context          TEXT NOT NULL DEFAULT '',    -- one-line situational description
    importance       INTEGER NOT NULL DEFAULT 5   -- 1-10, Generative-Agents-style
);

-- A-MEM (Phase 6): Zettelkasten-style semantic edges between facts.
-- Undirected: callers normalise so fact_a_id < fact_b_id at insert time.
CREATE TABLE IF NOT EXISTS semantic_fact_links (
    fact_a_id    INTEGER NOT NULL REFERENCES semantic_facts(id) ON DELETE CASCADE,
    fact_b_id    INTEGER NOT NULL REFERENCES semantic_facts(id) ON DELETE CASCADE,
    similarity   REAL NOT NULL,
    created_at   REAL NOT NULL,
    PRIMARY KEY (fact_a_id, fact_b_id)
);
CREATE INDEX IF NOT EXISTS idx_semantic_fact_links_a ON semantic_fact_links(fact_a_id, similarity DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_fact_links_b ON semantic_fact_links(fact_b_id, similarity DESC);
"""

# vec0 takes dims as a literal at CREATE time, so it has to be formatted
# separately and executed after sqlite_vec.load().
_VEC_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    chunk_id  INTEGER PRIMARY KEY,
    embedding FLOAT[{dims}] distance_metric=cosine
);

CREATE VIRTUAL TABLE IF NOT EXISTS skill_vec USING vec0(
    skill_id  INTEGER PRIMARY KEY,
    embedding FLOAT[{dims}] distance_metric=cosine
);

CREATE VIRTUAL TABLE IF NOT EXISTS trajectories_vec USING vec0(
    trajectory_id  INTEGER PRIMARY KEY,
    embedding      FLOAT[{dims}] distance_metric=cosine
);

CREATE VIRTUAL TABLE IF NOT EXISTS reflections_vec USING vec0(
    reflection_id  INTEGER PRIMARY KEY,
    embedding      FLOAT[{dims}] distance_metric=cosine
);

CREATE VIRTUAL TABLE IF NOT EXISTS semantic_facts_vec USING vec0(
    fact_id        INTEGER PRIMARY KEY,
    embedding      FLOAT[{dims}] distance_metric=cosine
);
"""


_MIGRATIONS = [
    # Phase 5: MemRL utility scores on reflections and trajectories.
    "ALTER TABLE reflections ADD COLUMN utility REAL NOT NULL DEFAULT 0.5",
    "ALTER TABLE trajectories ADD COLUMN utility REAL NOT NULL DEFAULT 0.5",
    # Phase 4.3: Episodic→semantic distillation output table.
    # CREATE IF NOT EXISTS is a no-op on fresh DBs (already in _SCHEMA);
    # on existing DBs without the table it creates it.
    """CREATE TABLE IF NOT EXISTS semantic_facts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    content          TEXT NOT NULL,
    source_chunk_ids TEXT NOT NULL,
    created_at       REAL NOT NULL
)""",
    # Phase 6: A-MEM note attributes on semantic_facts.
    "ALTER TABLE semantic_facts ADD COLUMN keywords TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE semantic_facts ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE semantic_facts ADD COLUMN context TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE semantic_facts ADD COLUMN importance INTEGER NOT NULL DEFAULT 5",
    # Phase 6: Zettelkasten-style edge table between facts.
    """CREATE TABLE IF NOT EXISTS semantic_fact_links (
    fact_a_id    INTEGER NOT NULL REFERENCES semantic_facts(id) ON DELETE CASCADE,
    fact_b_id    INTEGER NOT NULL REFERENCES semantic_facts(id) ON DELETE CASCADE,
    similarity   REAL NOT NULL,
    created_at   REAL NOT NULL,
    PRIMARY KEY (fact_a_id, fact_b_id)
)""",
    "CREATE INDEX IF NOT EXISTS idx_semantic_fact_links_a ON semantic_fact_links(fact_a_id, similarity DESC)",
    "CREATE INDEX IF NOT EXISTS idx_semantic_fact_links_b ON semantic_fact_links(fact_b_id, similarity DESC)",
    # Mirror an ON DELETE CASCADE into semantic_facts_vec — vec0 virtual
    # tables don't participate in FK cascades, so a trigger keeps the
    # vector table from accumulating orphans if a fact is ever deleted.
    # Placed in migrations (not inline _SCHEMA) because the referenced
    # virtual table is created in _VEC_SCHEMA, which runs after _SCHEMA.
    """CREATE TRIGGER IF NOT EXISTS semantic_facts_ad AFTER DELETE ON semantic_facts
       BEGIN DELETE FROM semantic_facts_vec WHERE fact_id = old.id; END""",
    # Phase 7 (RMM): citation feedback counters on reflections.
    # view_count = times this reflection was injected; cite_count = times the
    # next assistant response showed substantial token overlap with it.
    # Smoothed win-rate (cite_count + 1) / (view_count + 2) feeds retrieval.
    "ALTER TABLE reflections ADD COLUMN view_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE reflections ADD COLUMN cite_count INTEGER NOT NULL DEFAULT 0",
    # Phase 8: small KV metadata table (curator cooldown, etc).
    """CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)""",
]


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply additive ALTER TABLE migrations idempotently."""
    for sql in _MIGRATIONS:
        try:
            conn.execute(sql)
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


def _migrate_vec0_to_cosine(conn: sqlite3.Connection, target_dims: int) -> None:
    """Migrate vec0 tables to distance_metric=cosine if they were created without it.

    vec0 virtual tables cannot be ALTER TABLE'd — the only migration path is to
    read existing data, DROP the table, CREATE with the new metric, and re-INSERT.
    This is idempotent: tables that already have distance_metric=cosine are skipped.
    """
    tables = {
        "chunks_vec": ("chunk_id", "chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{dims}] distance_metric=cosine"),
        "skill_vec": ("skill_id", "skill_id INTEGER PRIMARY KEY, embedding FLOAT[{dims}] distance_metric=cosine"),
        "trajectories_vec": ("trajectory_id", "trajectory_id INTEGER PRIMARY KEY, embedding FLOAT[{dims}] distance_metric=cosine"),
        "reflections_vec": ("reflection_id", "reflection_id INTEGER PRIMARY KEY, embedding FLOAT[{dims}] distance_metric=cosine"),
    }
    for table, (pk_col, col_def) in tables.items():
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if row is None:
            continue  # table doesn't exist yet — _VEC_SCHEMA will create it correctly
        if "distance_metric=cosine" in (row[0] or ""):
            continue  # already correct
        log.info("migrating %s to distance_metric=cosine", table)
        existing = conn.execute(
            f"SELECT {pk_col}, embedding FROM {table}"
        ).fetchall()
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(
            f"CREATE VIRTUAL TABLE {table} USING vec0({col_def.format(dims=target_dims)})"
        )
        if existing:
            conn.executemany(
                f"INSERT INTO {table} ({pk_col}, embedding) VALUES (?, ?)",
                existing,
            )
        conn.commit()
        log.info("migrated %s: %d rows preserved", table, len(existing))


def open_db(
    workspace: Path, target_dims: int, busy_timeout_ms: int = 10000
) -> sqlite3.Connection:
    path = state_db(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    # Wait out a background VACUUM/purge lock instead of immediately raising
    # SQLITE_BUSY (which would silently drop the turn's archive write).
    conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(_SCHEMA)
    conn.executescript(_VEC_SCHEMA.format(dims=target_dims))
    _migrate_vec0_to_cosine(conn, target_dims)
    _apply_migrations(conn)
    conn.commit()
    return conn


def purge_older_than(conn: sqlite3.Connection, days: int) -> dict[str, int]:
    """Drop sessions and trajectories older than N days.

    Chunks and reflections follow sessions via ``ON DELETE CASCADE``,
    but vec0 virtual tables do NOT participate in cascades — we must
    collect IDs and clean up ``chunks_vec``, ``reflections_vec``, and
    ``trajectories_vec`` manually before the parent rows disappear.

    Returns ``{"sessions": n, "trajectories": n}`` for logging.
    """
    import time as _time  # noqa: PLC0415
    cutoff = _time.time() - days * 86400

    # Collect vec IDs BEFORE cascade deletes remove the parent rows.
    old_session_ids = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM sessions WHERE ended_at IS NOT NULL AND ended_at < ?",
            (cutoff,),
        ).fetchall()
    ]
    if old_session_ids:
        ph = ",".join("?" * len(old_session_ids))
        old_chunk_ids = [
            r[0]
            for r in conn.execute(
                f"SELECT id FROM chunks WHERE session_id IN ({ph})",
                old_session_ids,
            ).fetchall()
        ]
        old_ref_ids = [
            r[0]
            for r in conn.execute(
                f"SELECT id FROM reflections WHERE session_id IN ({ph})",
                old_session_ids,
            ).fetchall()
        ]
    else:
        old_chunk_ids = []
        old_ref_ids = []

    sess = conn.execute(
        "DELETE FROM sessions WHERE ended_at IS NOT NULL AND ended_at < ?",
        (cutoff,),
    ).rowcount

    # Clean up orphaned vec0 rows for chunks and reflections (batched).
    if old_chunk_ids:
        ph = ",".join("?" * len(old_chunk_ids))
        conn.execute(f"DELETE FROM chunks_vec WHERE chunk_id IN ({ph})", old_chunk_ids)
    if old_ref_ids:
        ph = ",".join("?" * len(old_ref_ids))
        conn.execute(
            f"DELETE FROM reflections_vec WHERE reflection_id IN ({ph})", old_ref_ids
        )

    # Collect trajectory IDs before deleting so we can clean up vec rows.
    old_traj_ids = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM trajectories WHERE created_at < ?",
            (cutoff,),
        ).fetchall()
    ]
    traj = conn.execute(
        "DELETE FROM trajectories WHERE created_at < ?",
        (cutoff,),
    ).rowcount
    if old_traj_ids:
        ph = ",".join("?" * len(old_traj_ids))
        conn.execute(
            f"DELETE FROM trajectories_vec WHERE trajectory_id IN ({ph})", old_traj_ids
        )
    conn.commit()
    return {"sessions": sess, "trajectories": traj}
