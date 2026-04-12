"""Session archive: SQLite + FTS5 + sqlite-vec, with hybrid search."""
from .archiver import SessionArchiver
from .db import open_db, purge_older_than
from .search import Hit, SessionSearchTool, hybrid_search, reciprocal_rank_fusion

__all__ = [
    "SessionArchiver",
    "open_db",
    "purge_older_than",
    "SessionSearchTool",
    "hybrid_search",
    "reciprocal_rank_fusion",
    "Hit",
]
