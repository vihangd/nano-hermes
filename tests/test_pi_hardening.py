"""Pi perf hardening: fact-table indexes, heavy-IO lock, snapshot retain."""
from __future__ import annotations

import asyncio

import nano_hermes


def test_semantic_facts_indexes_created(loop):
    hook = nano_hermes.install(loop)
    names = {
        r[0]
        for r in hook.db.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    }
    assert "idx_semantic_facts_created" in names
    assert "idx_semantic_facts_evict" in names


def test_index_migration_is_idempotent(loop):
    # install() runs the migration list; the IF NOT EXISTS indexes must not
    # raise on a second apply.
    hook = nano_hermes.install(loop)
    from nano_hermes.session.db import _apply_migrations

    _apply_migrations(hook.db)  # second pass — no error


def test_heavy_io_lock_is_present(loop):
    hook = nano_hermes.install(loop)
    assert isinstance(hook._heavy_io_lock, asyncio.Lock)


def test_snapshot_retain_default_is_small(loop):
    hook = nano_hermes.install(loop)
    # Whole-DB copies — kept few on an SD card.
    assert hook.config.skill_stats.snapshot_retain == 3
