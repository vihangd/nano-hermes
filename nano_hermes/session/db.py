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

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Callable

import sqlite_vec

from ..paths import state_db

log = logging.getLogger(__name__)


def _main_db_path(conn: sqlite3.Connection) -> str | None:
    """File path backing ``conn``'s main database, or ``None`` if it has no
    file (``:memory:`` / temp), which can't be reopened from another thread."""
    try:
        for _seq, name, file in conn.execute("PRAGMA database_list"):
            if name == "main":
                return file or None
    except sqlite3.Error:
        return None
    return None


async def run_vec_write(
    conn: sqlite3.Connection,
    fn: Callable[[sqlite3.Connection], object],
    *,
    busy_timeout_ms: int = 10000,
) -> None:
    """Run a vec-table write off the event loop on a short-lived connection.

    Background ``_embed_and_write`` tasks call this so the synchronous commit
    — and any WAL-checkpoint fsync it triggers — runs on a worker thread via
    ``asyncio.to_thread`` instead of stalling the single event loop on slow
    (SD-card) I/O. Mirrors the short-lived-connection pattern in
    ``NanoHermesHook._background_purge`` and never shares the loop-owned
    connection across threads.

    Falls back to a synchronous write on ``conn`` when the database has no
    backing file (e.g. ``:memory:``), which cannot be reopened elsewhere.
    """
    db_path = _main_db_path(conn)
    if not db_path or db_path == ":memory:":
        fn(conn)
        conn.commit()
        return

    def _work() -> None:
        w = sqlite3.connect(db_path)
        try:
            w.execute("PRAGMA foreign_keys = ON")
            w.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
            w.enable_load_extension(True)
            sqlite_vec.load(w)
            w.enable_load_extension(False)
            fn(w)
            w.commit()
        finally:
            w.close()

    await asyncio.to_thread(_work)


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
    status          TEXT NOT NULL DEFAULT 'active',   -- draft | active | stale | deprecated
    use_count       INTEGER NOT NULL DEFAULT 0,
    success_count   INTEGER NOT NULL DEFAULT 0,
    last_used_at    REAL,
    provenance      TEXT,                              -- JSON list of session ids
    content_hash    TEXT,                              -- sha1 of (name + description)
    indexed_at      REAL,                              -- last time we embedded this skill
    origin          TEXT NOT NULL DEFAULT 'user',      -- 'agent' (propose_skill) | 'user' (everything else)
    pinned          INTEGER NOT NULL DEFAULT 0         -- 1 = user-exempted from auto-evolution
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

-- FTS5 lexical mirror of trajectory task text, for hybrid (BM25 + dense)
-- retrieval in trajectory_search. External content keyed to trajectories.id.
CREATE VIRTUAL TABLE IF NOT EXISTS trajectories_fts USING fts5(
    task,
    content='trajectories',
    content_rowid='id',
    tokenize='porter'
);

CREATE TRIGGER IF NOT EXISTS trajectories_ai AFTER INSERT ON trajectories BEGIN
    INSERT INTO trajectories_fts(rowid, task) VALUES (new.id, new.task);
END;

CREATE TRIGGER IF NOT EXISTS trajectories_ad AFTER DELETE ON trajectories BEGIN
    INSERT INTO trajectories_fts(trajectories_fts, rowid, task) VALUES ('delete', old.id, old.task);
END;

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

-- principles_fts is a standalone (non-external-content) fts5 table, so a
-- delete is a plain DELETE by rowid — NOT the external-content 'delete'
-- command (that form raises "no row with rowid" on a standalone table).
CREATE TRIGGER IF NOT EXISTS principles_ad AFTER DELETE ON principles BEGIN
    DELETE FROM principles_fts WHERE rowid = old.id;
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
    importance       INTEGER NOT NULL DEFAULT 5,  -- 1-10, Generative-Agents-style
    -- Bi-temporal invalidation (Zep/Graphiti variant): NULL = currently true;
    -- a timestamp = the moment a newer fact superseded/contradicted this one.
    -- Non-destructive — the row is kept for history, filtered from live views.
    invalid_at       REAL
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

CREATE VIRTUAL TABLE IF NOT EXISTS principles_vec USING vec0(
    principle_id   INTEGER PRIMARY KEY,
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
    # Phase 9 (bi-temporal): supersession invalidation timestamp.
    # NULL = currently valid; set when a newer fact contradicts this one.
    "ALTER TABLE semantic_facts ADD COLUMN invalid_at REAL",
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
    # Phase 10: skill write-origin + pinning. Only 'agent'-authored skills
    # (created via propose_skill) are eligible for auto-evolution; everything
    # else — builtin, external, and hand-authored workspace skills — defaults
    # to 'user' and is protected. `pinned` lets a user exempt any skill.
    # Existing rows migrate to 'user', which conservatively shields skills the
    # agent proposed before this column existed; newly proposed skills get
    # 'agent' going forward.
    "ALTER TABLE skill_stats ADD COLUMN origin TEXT NOT NULL DEFAULT 'user'",
    "ALTER TABLE skill_stats ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0",
    # Phase 11 (ACE delta-playbook): automatic principle evolution.
    # success_count is the 'helpful' counter; add 'harmful' for failed-session
    # attribution. origin distinguishes manual (principle_tool) from auto
    # (curator) rows so pruning only ever removes curator-authored principles.
    "ALTER TABLE principles ADD COLUMN harmful_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE principles ADD COLUMN origin TEXT NOT NULL DEFAULT 'agent'",
    "ALTER TABLE principles ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE principles ADD COLUMN updated_at REAL",
    # FK-less cascade into the vec0 shadow table (vec0 ignores FK cascades).
    # WHEN-guarded: vec0 raises on DELETE of a missing rowid, and FTS-only
    # principles (embedding unavailable at write time) have no vec row.
    """CREATE TRIGGER IF NOT EXISTS principles_ad_vec AFTER DELETE ON principles
       WHEN EXISTS (SELECT 1 FROM principles_vec WHERE principle_id = old.id)
       BEGIN DELETE FROM principles_vec WHERE principle_id = old.id; END""",
    # Replace the original principles_ad trigger, which used the external-content
    # 'delete' command on a standalone fts5 table and raised on real deletes
    # (dormant until the curator introduced principle deletion).
    "DROP TRIGGER IF EXISTS principles_ad",
    """CREATE TRIGGER principles_ad AFTER DELETE ON principles BEGIN
       DELETE FROM principles_fts WHERE rowid = old.id; END""",
    # Serve the contradiction-sweep anchor scan (ORDER BY created_at DESC) and
    # the fact-eviction scan (ORDER BY importance, created_at) without a full
    # table sort once semantic_facts grows. Partial on the live-fact predicate.
    "CREATE INDEX IF NOT EXISTS idx_semantic_facts_created "
    "ON semantic_facts(created_at) WHERE invalid_at IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_semantic_facts_evict "
    "ON semantic_facts(importance, created_at) WHERE invalid_at IS NULL",
    # Phase 12 (write-approval gate): pending store for autonomous evolution
    # writes staged under write_approval='approve'. payload is JSON — the new
    # SKILL.md body (skills) or the curator op list (principles). base_hash is
    # the sha256 of the live SKILL.md at stage time, used to refuse a stale
    # replay (skills only). status: pending | approved | rejected | stale.
    """CREATE TABLE IF NOT EXISTS pending_writes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    subsystem    TEXT NOT NULL,                       -- 'skills' | 'principles'
    skill_name   TEXT,                                -- NULL for principle ops
    payload      TEXT NOT NULL,                        -- JSON
    base_hash    TEXT,                                -- sha256 of live body at stage (skills)
    reason       TEXT,
    origin       TEXT NOT NULL DEFAULT 'background',   -- 'rewriter'|'gepa'|'umbrella'|'curator'
    status       TEXT NOT NULL DEFAULT 'pending',      -- pending|approved|rejected|stale
    created_at   REAL NOT NULL,
    resolved_at  REAL
)""",
    "CREATE INDEX IF NOT EXISTS idx_pending_open "
    "ON pending_writes(subsystem) WHERE status='pending'",
    # Phase 13 (Dynamic Cheatsheet): per-session transferable lesson extraction.
    # fact_type distinguishes cheatsheet lessons ('cheatsheet') from distilled
    # semantic facts ('fact'). task_category indexes by task for KNN retrieval.
    "ALTER TABLE semantic_facts ADD COLUMN fact_type TEXT NOT NULL DEFAULT 'fact'",
    "ALTER TABLE semantic_facts ADD COLUMN task_category TEXT",
    "CREATE INDEX IF NOT EXISTS idx_semantic_facts_type "
    "ON semantic_facts(fact_type) WHERE invalid_at IS NULL",
    # Phase 14 (OPRO): prompt variants and scores for meta-optimization.
    """CREATE TABLE IF NOT EXISTS prompt_versions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_name  TEXT NOT NULL,
    body         TEXT NOT NULL,
    score        REAL,
    active       INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,
    scored_at    REAL
)""",
    "CREATE INDEX IF NOT EXISTS idx_prompt_versions_name "
    "ON prompt_versions(prompt_name, active)",
    # Phase 15 (auto trust-scoring): mutable per-fact reliability, moved by the
    # session outcome of injected lessons. Neutral 1.0 = unrated (distance/trust
    # unchanged). Only cheatsheet/expel facts are injected, so others stay 1.0.
    "ALTER TABLE semantic_facts ADD COLUMN trust_score REAL NOT NULL DEFAULT 1.0",
    # Phase 15 (skill telemetry): track context-load activity, not just formal
    # use, so a frequently-viewed skill isn't staled as if dormant.
    "ALTER TABLE skill_stats ADD COLUMN last_viewed_at REAL",
    "ALTER TABLE skill_stats ADD COLUMN view_count INTEGER NOT NULL DEFAULT 0",
]


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply additive migrations idempotently, gated by PRAGMA user_version."""
    # _MIGRATIONS is append-only, so its length is a monotonic schema version.
    target = len(_MIGRATIONS)
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current >= target:
        return
    clean = True
    for sql in _MIGRATIONS:
        try:
            conn.execute(sql)
            conn.commit()
        except sqlite3.OperationalError as e:
            # "duplicate column" / "already exists" are expected on a legacy DB
            # whose rows predate the version stamp — benign. Anything else (e.g.
            # "database is locked") means this column may not exist; don't stamp,
            # so the next open retries instead of silently skipping it forever.
            if "duplicate column" in str(e) or "already exists" in str(e):
                continue
            log.warning("migration failed, will retry next open: %s", e)
            clean = False
    if clean:
        conn.execute(f"PRAGMA user_version = {target}")
        conn.commit()


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
    conn.execute("PRAGMA cache_size = -8000")    # 8 MB page cache — cuts SD card thrash on Pi
    conn.execute("PRAGMA temp_store = MEMORY")   # avoid tmp-file I/O for sort/index ops
    conn.execute("PRAGMA mmap_size = 67108864")  # 64 MB mmap window for read-heavy queries
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
    _backfill_trajectories_fts(conn)
    # Self-heal any index a previous run flagged as corrupt. Only does work
    # when a marker is actually set, so a healthy DB pays one indexed lookup.
    rebuild_stale_fts(conn)
    conn.commit()
    return conn


_FTS_BACKFILL_FLAG = "trajectories_fts_backfilled"


def _backfill_trajectories_fts(conn: sqlite3.Connection) -> None:
    """One-time populate of trajectories_fts for DBs created before it existed.

    The table and triggers are in _SCHEMA (idempotent), but triggers only fire
    on future inserts, so a pre-existing DB has trajectory rows with an empty
    FTS index. We can't detect that via ``COUNT(*)`` — on an external-content
    FTS5 table that reads the content (trajectories) table, not the index — so
    we gate the one-time rebuild on a ``meta`` flag instead. On a fresh DB the
    rebuild is a harmless no-op (no rows yet).
    """
    try:
        done = conn.execute(
            "SELECT 1 FROM meta WHERE key = ?", (_FTS_BACKFILL_FLAG,)
        ).fetchone()
        if done:
            return
        conn.execute("INSERT INTO trajectories_fts(trajectories_fts) VALUES('rebuild')")
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, '1') "
            "ON CONFLICT(key) DO NOTHING",
            (_FTS_BACKFILL_FLAG,),
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass  # table missing on a partially-migrated DB — next open retries


# ---------------------------------------------------------------------------
# FTS health
# ---------------------------------------------------------------------------

FTS_TABLES = ("chunks_fts", "trajectories_fts", "principles_fts")
_FTS_STALE_PREFIX = "fts_stale:"

# A malformed MATCH expression and a corrupt index both surface as
# OperationalError. Only the latter is worth remembering: the query errors are
# per-query and self-correcting, while corruption silently degrades hybrid
# retrieval to vector-only forever.
_FTS_CORRUPTION_MARKERS = ("malformed", "corrupt", "disk image", "no such table")


def is_fts_corruption(exc: BaseException) -> bool:
    """Whether *exc* looks like index corruption rather than a bad query."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _FTS_CORRUPTION_MARKERS)


def mark_fts_stale(conn: sqlite3.Connection, table: str) -> None:
    """Record that *table*'s FTS index is untrustworthy. Best-effort.

    Called from a read path that has just seen a database error, so the
    connection may itself be unhealthy — never raise from here.
    """
    try:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, '1') "
            "ON CONFLICT(key) DO NOTHING",
            (f"{_FTS_STALE_PREFIX}{table}",),
        )
        conn.commit()
    except sqlite3.Error:
        log.debug("could not record FTS stale marker for %s", table, exc_info=True)


def stale_fts_tables(conn: sqlite3.Connection) -> list[str]:
    """Return the FTS tables currently flagged stale."""
    try:
        rows = conn.execute(
            "SELECT key FROM meta WHERE key LIKE ?", (f"{_FTS_STALE_PREFIX}%",)
        ).fetchall()
    except sqlite3.Error:
        return []
    return sorted(r[0][len(_FTS_STALE_PREFIX):] for r in rows)


def _clear_fts_marker(conn: sqlite3.Connection, table: str) -> None:
    """Remove one stale marker. Best-effort, mirrors mark_fts_stale."""
    try:
        conn.execute(
            "DELETE FROM meta WHERE key = ?", (f"{_FTS_STALE_PREFIX}{table}",)
        )
        conn.commit()
    except sqlite3.Error:
        log.debug("could not clear FTS stale marker for %s", table, exc_info=True)


def rebuild_stale_fts(conn: sqlite3.Connection) -> list[str]:
    """Rebuild any FTS index flagged stale; clear the flag on success.

    Uses FTS5's native ``'rebuild'`` command, which repopulates the index from
    the content table in one statement — no chunked re-indexing needed. Returns
    the tables successfully rebuilt.
    """
    repaired: list[str] = []
    for table in stale_fts_tables(conn):
        if table not in FTS_TABLES:
            # Never interpolate an unrecognised name into SQL — and drop the
            # marker rather than leaving it to report a degradation forever
            # that no rebuild can clear.
            _clear_fts_marker(conn, table)
            log.warning("dropped unrecognised FTS stale marker: %s", table)
            continue
        try:
            conn.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")
            conn.execute(
                "DELETE FROM meta WHERE key = ?", (f"{_FTS_STALE_PREFIX}{table}",)
            )
            conn.commit()
            repaired.append(table)
            log.info("rebuilt stale FTS index: %s", table)
        except sqlite3.Error:
            # Leave the flag set so the next open retries and nano_status
            # keeps reporting the degradation.
            log.warning("FTS rebuild failed for %s", table, exc_info=True)
    return repaired


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

    # vec0 virtual tables don't participate in FK cascades, so their orphan
    # rows must be deleted explicitly. Use nested subqueries (not a
    # Python-materialised ``IN (?, ?, …)`` list) so a large purge can't blow
    # past SQLite's SQLITE_MAX_VARIABLE_NUMBER cap (as low as 32766 on some
    # builds) — which would raise OperationalError, get swallowed by the
    # background purge, and silently stop both the purge and its VACUUM,
    # letting the DB grow without bound on a swapless Pi.
    #
    # The vec0 cleanups must run BEFORE the parent DELETE: once sessions are
    # gone their chunks/reflections cascade away and the subqueries find
    # nothing.
    _old_sessions = (
        "SELECT id FROM sessions WHERE ended_at IS NOT NULL AND ended_at < ?"
    )
    conn.execute(
        f"DELETE FROM chunks_vec WHERE chunk_id IN ("
        f"  SELECT id FROM chunks WHERE session_id IN ({_old_sessions}))",
        (cutoff,),
    )
    conn.execute(
        f"DELETE FROM reflections_vec WHERE reflection_id IN ("
        f"  SELECT id FROM reflections WHERE session_id IN ({_old_sessions}))",
        (cutoff,),
    )
    sess = conn.execute(
        "DELETE FROM sessions WHERE ended_at IS NOT NULL AND ended_at < ?",
        (cutoff,),
    ).rowcount

    conn.execute(
        "DELETE FROM trajectories_vec WHERE trajectory_id IN ("
        "  SELECT id FROM trajectories WHERE created_at < ?)",
        (cutoff,),
    )
    traj = conn.execute(
        "DELETE FROM trajectories WHERE created_at < ?",
        (cutoff,),
    ).rowcount
    conn.commit()
    return {"sessions": sess, "trajectories": traj}


def evict_low_value_facts(
    conn: sqlite3.Connection,
    *,
    retention_days: int,
    importance_floor: int,
    superseded_grace_days: int,
    max_per_run: int,
    trust_floor: float = 0.0,
    now: float | None = None,
) -> int:
    """Bound the otherwise-unbounded ``semantic_facts`` table.

    Two conservative tiers, each capped at ``max_per_run`` deletions total:
      1. Superseded facts (``invalid_at`` set) older than the grace window —
         pure dead weight kept only briefly for audit.
      2. *Valid* facts that are both older than ``retention_days`` AND below
         ``importance_floor`` (the distiller's 1–10 score). High-importance
         facts never auto-evict, regardless of age.

    Deletion is ordered lowest-value-first and bounded by a subquery LIMIT
    (SQLite's ``DELETE … LIMIT`` is a non-default compile option). The
    ``semantic_facts_ad`` trigger removes the matching ``semantic_facts_vec``
    rows and ``semantic_fact_links`` cascade via FK, so callers need only
    delete the parent rows. Returns the number of facts removed.
    """
    import time as _time  # noqa: PLC0415
    now_ts = now if now is not None else _time.time()
    removed = 0

    if superseded_grace_days >= 0 and max_per_run > 0:
        cutoff = now_ts - superseded_grace_days * 86400
        removed += conn.execute(
            "DELETE FROM semantic_facts WHERE id IN ("
            "  SELECT id FROM semantic_facts "
            "  WHERE invalid_at IS NOT NULL AND invalid_at < ? "
            "  ORDER BY invalid_at ASC LIMIT ?)",
            (cutoff, max_per_run),
        ).rowcount

    remaining = max_per_run - removed
    if retention_days > 0 and remaining > 0:
        cutoff = now_ts - retention_days * 86400
        # Evict old facts that are either low-importance OR distrusted (a lesson
        # driven below the retrieval floor is hidden anyway — reclaim its disk).
        removed += conn.execute(
            "DELETE FROM semantic_facts WHERE id IN ("
            "  SELECT id FROM semantic_facts "
            "  WHERE invalid_at IS NULL AND created_at < ? "
            "    AND (importance < ? OR trust_score < ?) "
            "  ORDER BY trust_score ASC, importance ASC, created_at ASC LIMIT ?)",
            (cutoff, importance_floor, trust_floor, remaining),
        ).rowcount

    conn.commit()
    return removed
