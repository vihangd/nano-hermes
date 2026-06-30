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
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import aiohttp

from ..config import EmbeddingConfig, EmbeddingProvider

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

def _embeddings_in_index_order(data: dict) -> list[list[float]]:
    """Extract embeddings from an OpenAI-compatible response in input order.

    The response carries a per-row ``index`` because array order is not
    contractual. Batch callers bind vectors positionally (zip with chunk_ids /
    skill names), so a provider that returns rows out of order would silently
    misbind every vector to the wrong text. Sort by ``index`` to prevent that.
    """
    rows = sorted(data["data"], key=lambda row: row.get("index", 0))
    return [row["embedding"] for row in rows]


# OpenAI-compatible embedding endpoints.
_ENDPOINTS: dict[str, str] = {
    "deepinfra":  "https://api.deepinfra.com/v1/openai/embeddings",
    "together":   "https://api.together.xyz/v1/embeddings",
    "openrouter": "https://openrouter.ai/api/v1/embeddings",
}

# Per-process unhealthy-provider cache.
# Key: (provider_name, model). Value: monotonic time after which to retry.
_unhealthy_until: dict[tuple[str, str], float] = {}
# Throttle "skipping unhealthy provider" log to once per minute per provider.
_last_skip_logged: dict[tuple[str, str], float] = {}

_UNHEALTHY_TTL = 600.0       # 10 min default; configurable in EmbeddingConfig
_SKIP_LOG_THROTTLE = 60.0    # 1 min between skip-log lines per provider

# Shared TCP connector — keeps connections alive across embed calls so we don't
# pay TLS handshake overhead (50-300ms on ARM) on every request.
_shared_connector: aiohttp.TCPConnector | None = None


def _get_connector() -> aiohttp.TCPConnector:
    global _shared_connector
    if _shared_connector is None or _shared_connector.closed:
        _shared_connector = aiohttp.TCPConnector(
            limit=4,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
    return _shared_connector


class AllProvidersFailed(RuntimeError):
    pass


@dataclass
class EmbeddingChain:
    config: EmbeddingConfig
    _session: aiohttp.ClientSession | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> "EmbeddingChain":
        self._session = aiohttp.ClientSession(
            connector=_get_connector(),
            connector_owner=False,  # don't close the shared connector on exit
            timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
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
        ttl = getattr(self.config, "unhealthy_ttl_seconds", _UNHEALTHY_TTL)
        now = time.monotonic()
        for p in self.config.chain:
            key = (p.provider, self.config.model)
            if _unhealthy_until.get(key, 0.0) > now:
                last_log = _last_skip_logged.get(key, 0.0)
                if now - last_log > _SKIP_LOG_THROTTLE:
                    log.debug("skipping unhealthy provider %s (cached)", p.provider)
                    _last_skip_logged[key] = now
                errors.append(f"{p.provider}: unhealthy (cached)")
                continue
            try:
                raws = await self._call(p, list(texts))
                # Successful call — clear any stale unhealthy entry.
                _unhealthy_until.pop(key, None)
                return [self._postprocess(v) for v in raws]
            except aiohttp.ClientResponseError as e:
                if e.status in (402, 429) or e.status >= 500:
                    _unhealthy_until[key] = now + ttl
                    log.warning(
                        "embedding provider %s returned HTTP %d — marked unhealthy for %.0fs",
                        p.provider, e.status, ttl,
                    )
                else:
                    log.warning("embedding provider %s failed: %s", p.provider, e)
                errors.append(f"{p.provider}: HTTP {e.status}")
            except Exception as e:
                errors.append(f"{p.provider}: {e}")
                log.warning("embedding provider %s failed: %s", p.provider, e)
        raise AllProvidersFailed("; ".join(errors))

    async def _call(  # noqa: D401
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
        return _embeddings_in_index_order(data)

    def _postprocess(self, raw: list[float]) -> np.ndarray:
        import numpy as np  # noqa: PLC0415
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


