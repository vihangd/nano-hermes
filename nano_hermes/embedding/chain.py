"""Same-model embedding chain with hot failover.

All providers serve the same open-weight model (``BAAI/bge-m3``), so vectors
are interchangeable across providers for both reads and writes. On error or
timeout, the chain walks forward and retries the next provider. If every
provider fails, ``AllProvidersFailed`` is raised and the caller degrades to
FTS5-only retrieval.

Writes buffer a 512-dim truncated, L2-normalized vector. Storage per year on
a Pi at ~7k chunks is ~14 MB — fits SD cards comfortably.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Sequence

import aiohttp
import numpy as np

from ..config import EmbeddingConfig, EmbeddingProvider

log = logging.getLogger(__name__)

# OpenAI-compatible embedding endpoints.
_ENDPOINTS: dict[str, str] = {
    "deepinfra":  "https://api.deepinfra.com/v1/openai/embeddings",
    "together":   "https://api.together.xyz/v1/embeddings",
    "openrouter": "https://openrouter.ai/api/v1/embeddings",
}


class AllProvidersFailed(RuntimeError):
    pass


@dataclass
class EmbeddingChain:
    config: EmbeddingConfig
    _session: aiohttp.ClientSession | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> "EmbeddingChain":
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        )
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Embed a batch. First working provider wins; fail over on errors."""
        if not texts:
            return []
        errors: list[str] = []
        for p in self.config.chain:
            try:
                raws = await self._call(p, list(texts))
                return [self._postprocess(v) for v in raws]
            except Exception as e:  # aiohttp errors, KeyError on payload, RuntimeError
                errors.append(f"{p.provider}: {e}")
                log.warning("embedding provider %s failed: %s", p.provider, e)
        raise AllProvidersFailed("; ".join(errors))

    async def _call(
        self, p: EmbeddingProvider, texts: list[str]
    ) -> list[list[float]]:
        assert self._session is not None, "use `async with` to open the chain"
        api_key = os.environ.get(p.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing env var {p.api_key_env}")
        url = p.base_url or _ENDPOINTS[p.provider]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.config.model, "input": texts}
        async with self._session.post(url, headers=headers, json=payload) as r:
            r.raise_for_status()
            data = await r.json()
        return [row["embedding"] for row in data["data"]]

    def _postprocess(self, raw: list[float]) -> np.ndarray:
        v = np.asarray(raw, dtype=np.float32)
        if v.shape[0] != self.config.native_dims:
            raise ValueError(
                f"expected {self.config.native_dims} dims, got {v.shape[0]}"
            )
        v = v[: self.config.target_dims]
        if self.config.normalize_after_truncate:
            n = float(np.linalg.norm(v))
            if n > 0.0:
                v = v / n
        return v


