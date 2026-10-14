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
import asyncio
import os
import ssl
import time
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import httpx

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

# One pooled client per event loop — keeps connections alive across embed
# calls so we don't pay a TLS handshake (50-300 ms on ARM) on every request.
# httpx, not aiohttp: nanobot already loads httpx, while aiohttp cost this
# plugin +12 MB RSS and ~56 ms of import (about 1.5 s on a Pi 3B+) for the
# same job. Per loop because a client's connections belong to the loop that
# opened them; the loop is held weakly so a finished one can be collected.
#
# Settings that keep aiohttp's behaviour where httpx's defaults differ:
# - keepalive_expiry: httpx drops idle connections after 5 s, which at one
#   embed per turn means a fresh TLS handshake nearly every call. 30 s stays
#   under common server idle timeouts (nginx 75 s, most LBs 60 s).
# - follow_redirects: a custom base_url behind a 307/308 kept working.
# - trust_env=False: embedding traffic did not go through HTTP(S)_PROXY /
#   ALL_PROXY, and a SOCKS proxy without socksio would fail client creation.
# - _SSL: the system trust store (honours SSL_CERT_FILE / SSL_CERT_DIR);
#   httpx with trust_env=False would trust only certifi's bundle. Built at
#   import, like aiohttp did, so the first embed doesn't load CA certs on the
#   event loop mid-turn.
# ponytail: no DNS cache (aiohttp had a 300 s one); a turn more than 30 s
# after the last pays a lookup. Add a resolver cache if a Pi measures it.
_LIMITS = httpx.Limits(
    max_connections=4, max_keepalive_connections=4, keepalive_expiry=30.0
)
_SSL = ssl.create_default_context()
_shared_client: tuple["weakref.ref[asyncio.AbstractEventLoop]", httpx.AsyncClient] | None = None


def _get_client() -> httpx.AsyncClient:
    global _shared_client
    loop = asyncio.get_running_loop()
    if (
        _shared_client is None
        or _shared_client[0]() is not loop
        or _shared_client[1].is_closed
    ):
        # A client left from a finished loop can't be awaited closed from here;
        # its sockets go with it when it is collected.
        _shared_client = (
            weakref.ref(loop),
            httpx.AsyncClient(
                limits=_LIMITS, verify=_SSL, follow_redirects=True, trust_env=False
            ),
        )
    return _shared_client[1]


class AllProvidersFailed(RuntimeError):
    pass


@dataclass
class EmbeddingChain:
    config: EmbeddingConfig
    _session: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> "EmbeddingChain":
        self._session = _get_client()
        return self

    async def __aexit__(self, *_exc) -> None:
        # The client is shared; leaving the block only releases this handle.
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
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in (402, 429) or status >= 500:
                    _unhealthy_until[key] = now + ttl
                    log.warning(
                        "embedding provider %s returned HTTP %d — marked unhealthy for %.0fs",
                        p.provider, status, ttl,
                    )
                else:
                    log.warning("embedding provider %s failed: %s", p.provider, e)
                errors.append(f"{p.provider}: HTTP {status}")
            except (asyncio.TimeoutError, httpx.TimeoutException) as e:
                # A hung provider would otherwise cost every turn the full
                # timeout before failover.
                _unhealthy_until[key] = now + ttl
                log.warning(
                    "embedding provider %s timed out — marked unhealthy for %.0fs",
                    p.provider, ttl,
                )
                errors.append(f"{p.provider}: timeout ({type(e).__name__})")
            except Exception as e:
                errors.append(f"{p.provider}: {e!r}")
                log.warning("embedding provider %s failed: %r", p.provider, e)
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
        # A bare httpx timeout bounds each phase (connect, each read, pool
        # wait) separately, so a slow-dripping provider could hold a call for
        # several times timeout_seconds; aiohttp's was a total. wait_for
        # restores the cap — this call can sit on the turn path (cheatsheet,
        # trajectory and global-reflection injection, session_search).
        r = await asyncio.wait_for(
            self._session.post(
                url, headers=headers, json=payload, timeout=self.config.timeout_seconds
            ),
            timeout=self.config.timeout_seconds,
        )
        r.raise_for_status()
        return _embeddings_in_index_order(r.json())

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


