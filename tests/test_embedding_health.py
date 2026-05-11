"""Tests for per-process unhealthy-provider cache in EmbeddingChain."""
from __future__ import annotations

import time
from unittest.mock import patch

import aiohttp
import pytest

from nano_hermes.embedding.chain import (
    EmbeddingChain,
    _unhealthy_until,
    _last_skip_logged,
)
from nano_hermes.config import EmbeddingConfig, EmbeddingProvider


def _make_config(ttl: float = 600.0) -> EmbeddingConfig:
    return EmbeddingConfig(
        model="test-model",
        native_dims=4,
        target_dims=4,
        unhealthy_ttl_seconds=ttl,
        chain=[
            EmbeddingProvider(provider="deepinfra", api_key_env="DEEPINFRA_API_KEY"),
            EmbeddingProvider(provider="together", api_key_env="TOGETHER_API_KEY"),
        ],
    )


def _make_402_error() -> aiohttp.ClientResponseError:
    req_info = aiohttp.RequestInfo(
        url=aiohttp.typedefs.URL("http://x"),
        method="POST",
        headers=None,
        real_url=aiohttp.typedefs.URL("http://x"),
    )
    return aiohttp.ClientResponseError(req_info, history=(), status=402)


@pytest.fixture(autouse=True)
def clear_cache():
    """Reset module-level cache before each test."""
    _unhealthy_until.clear()
    _last_skip_logged.clear()
    yield
    _unhealthy_until.clear()
    _last_skip_logged.clear()


class TestUnhealthyProviderCache:
    async def test_provider_marked_unhealthy_on_402(self, monkeypatch):
        """After a 402, the provider is skipped on the next call."""
        cfg = _make_config(ttl=600.0)
        chain = EmbeddingChain(cfg)

        call_count = {"deepinfra": 0, "together": 0}

        async def fake_call(p, texts):
            call_count[p.provider] += 1
            if p.provider == "deepinfra":
                raise _make_402_error()
            return [[1.0, 0.0, 0.0, 0.0]]

        monkeypatch.setenv("DEEPINFRA_API_KEY", "x")
        monkeypatch.setenv("TOGETHER_API_KEY", "y")

        async with chain:
            with patch.object(chain, "_call", side_effect=fake_call):
                result = await chain.embed(["hello"])

        assert result is not None
        assert call_count["deepinfra"] == 1
        assert call_count["together"] == 1

        # Second call: deepinfra should be skipped entirely.
        call_count2 = {"deepinfra": 0, "together": 0}

        async def fake_call2(p, texts):
            call_count2[p.provider] += 1
            return [[1.0, 0.0, 0.0, 0.0]]

        async with chain:
            with patch.object(chain, "_call", side_effect=fake_call2):
                await chain.embed(["world"])

        assert call_count2["deepinfra"] == 0, "deepinfra should be skipped (unhealthy)"
        assert call_count2["together"] == 1

    async def test_ttl_expiry_allows_retry(self, monkeypatch):
        """After TTL expires, the provider is retried."""
        cfg = _make_config(ttl=0.001)  # 1ms TTL
        chain = EmbeddingChain(cfg)

        monkeypatch.setenv("DEEPINFRA_API_KEY", "x")
        monkeypatch.setenv("TOGETHER_API_KEY", "y")

        async def fail_deepinfra(p, texts):
            if p.provider == "deepinfra":
                raise _make_402_error()
            return [[1.0, 0.0, 0.0, 0.0]]

        async with chain:
            with patch.object(chain, "_call", side_effect=fail_deepinfra):
                await chain.embed(["first call marks deepinfra unhealthy"])

        key = ("deepinfra", "test-model")
        assert _unhealthy_until.get(key, 0.0) > 0.0, "should be marked unhealthy"

        # Wait past TTL.
        time.sleep(0.01)

        # Now deepinfra should be retried.
        call_count = {"deepinfra": 0}

        async def count_deepinfra(p, texts):
            if p.provider == "deepinfra":
                call_count["deepinfra"] += 1
            return [[1.0, 0.0, 0.0, 0.0]]

        async with chain:
            with patch.object(chain, "_call", side_effect=count_deepinfra):
                await chain.embed(["second call after TTL"])

        assert call_count["deepinfra"] == 1, "deepinfra should be retried after TTL"

    async def test_429_marks_unhealthy(self, monkeypatch):
        """HTTP 429 also marks the provider unhealthy."""
        cfg = _make_config()
        chain = EmbeddingChain(cfg)
        monkeypatch.setenv("DEEPINFRA_API_KEY", "x")
        monkeypatch.setenv("TOGETHER_API_KEY", "y")

        req_info = aiohttp.RequestInfo(
            url=aiohttp.typedefs.URL("http://x"),
            method="POST",
            headers=None,
            real_url=aiohttp.typedefs.URL("http://x"),
        )
        err_429 = aiohttp.ClientResponseError(req_info, history=(), status=429)

        async def fail_429(p, texts):
            if p.provider == "deepinfra":
                raise err_429
            return [[1.0, 0.0, 0.0, 0.0]]

        async with chain:
            with patch.object(chain, "_call", side_effect=fail_429):
                await chain.embed(["text"])

        key = ("deepinfra", "test-model")
        assert _unhealthy_until.get(key, 0.0) > time.monotonic() - 1.0

    async def test_non_rate_limit_error_does_not_mark_unhealthy(self, monkeypatch):
        """HTTP 400 (bad request) should NOT mark the provider unhealthy."""
        cfg = _make_config()
        chain = EmbeddingChain(cfg)
        monkeypatch.setenv("DEEPINFRA_API_KEY", "x")
        monkeypatch.setenv("TOGETHER_API_KEY", "y")

        req_info = aiohttp.RequestInfo(
            url=aiohttp.typedefs.URL("http://x"),
            method="POST",
            headers=None,
            real_url=aiohttp.typedefs.URL("http://x"),
        )
        err_400 = aiohttp.ClientResponseError(req_info, history=(), status=400)

        async def fail_400(p, texts):
            if p.provider == "deepinfra":
                raise err_400
            return [[1.0, 0.0, 0.0, 0.0]]

        async with chain:
            with patch.object(chain, "_call", side_effect=fail_400):
                await chain.embed(["text"])

        key = ("deepinfra", "test-model")
        assert key not in _unhealthy_until, "HTTP 400 should NOT mark provider unhealthy"

    async def test_successful_call_clears_unhealthy(self, monkeypatch):
        """A successful call clears the unhealthy cache entry."""
        cfg = _make_config()
        chain = EmbeddingChain(cfg)
        monkeypatch.setenv("DEEPINFRA_API_KEY", "x")

        # Manually seed the cache.
        key = ("deepinfra", "test-model")
        _unhealthy_until[key] = 0.0  # expired TTL — will retry

        async def succeed(p, texts):
            return [[1.0, 0.0, 0.0, 0.0]]

        async with chain:
            with patch.object(chain, "_call", side_effect=succeed):
                await chain.embed(["text"])

        assert key not in _unhealthy_until, "successful call should clear unhealthy entry"
