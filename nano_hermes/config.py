"""Pydantic config schema for nano-hermes.

Trimmed to only what we own. Paths are derived from ``loop.workspace``,
not configured here, so there's no overlap with nanobot's own config.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EmbeddingProvider(BaseModel):
    provider: Literal["deepinfra", "together", "openrouter"]
    api_key_env: str
    base_url: str | None = None  # override the default OpenAI-compat endpoint


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    native_dims: int = 1024
    target_dims: int = 512
    normalize_after_truncate: bool = True
    timeout_seconds: float = 20.0
    chain: list[EmbeddingProvider] = Field(
        default_factory=lambda: [
            EmbeddingProvider(provider="deepinfra", api_key_env="DEEPINFRA_API_KEY"),
            EmbeddingProvider(provider="together", api_key_env="TOGETHER_API_KEY"),
            EmbeddingProvider(provider="openrouter", api_key_env="OPENROUTER_API_KEY"),
        ]
    )


class MemoryBudgets(BaseModel):
    """Char budgets enforced by ``BudgetedMemory`` on writes.

    Nanobot's underlying MemoryStore does not enforce these — we do, on
    behalf of the agent's ``memory_patch`` tool. Writes from nanobot's
    Dream / Consolidator path bypass budgets intentionally: curated
    background writes get latitude.
    """
    memory_md_chars: int = 2200
    user_md_chars: int = 1375
    soul_md_chars: int = 1500


class RetrievalConfig(BaseModel):
    fts_k: int = 25       # FTS5 candidate pool
    vec_k: int = 25       # vector candidate pool
    rrf_k: int = 60       # RRF smoothing constant (Cormack et al. default)
    final_k: int = 8


class ReflectionConfig(BaseModel):
    """Reflexion nudge trigger + injection limits.

    See ``nano_hermes/reflect/salience.py`` for how the score is computed.
    Default threshold of 5.0 fires on ANY of: one error, one tool-burst
    iteration, three user-correction phrases, or a mix that adds up.
    """
    threshold: float = 5.0
    recent_limit: int = 5   # max reflections injected per iteration


class NanoHermesConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    memory: MemoryBudgets = Field(default_factory=MemoryBudgets)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    trajectory_retention_days: int = 45
    reflection_scope: Literal["session", "global"] = "session"
