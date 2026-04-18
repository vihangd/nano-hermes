"""Unit tests for nano_hermes.embedding.chain.EmbeddingChain.

These tests mock the per-provider HTTP call (`EmbeddingChain._call`) rather
than aiohttp internals — the failover *logic* is the load-bearing thing,
not the HTTP plumbing (which production exercises constantly).
"""
from __future__ import annotations

import numpy as np
import pytest

from nano_hermes.config import EmbeddingConfig, EmbeddingProvider
from nano_hermes.embedding.chain import AllProvidersFailed, EmbeddingChain


def _config(*provider_names: str, target_dims: int = 4) -> EmbeddingConfig:
    """Build a tiny EmbeddingConfig with N fake providers."""
    chain = [
        EmbeddingProvider(provider=name, api_key_env=f"FAKE_{name.upper()}_KEY")
        for name in provider_names
    ]
    return EmbeddingConfig(
        model="fake-model",
        native_dims=8,
        target_dims=target_dims,
        normalize_after_truncate=True,
        timeout_seconds=1.0,
        chain=chain,
    )


class TestPostprocess:
    def test_truncates_to_target_dims(self):
        chain = EmbeddingChain(config=_config("deepinfra", target_dims=4))
        v = chain._postprocess([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        assert v.shape == (4,)

    def test_l2_normalizes_after_truncation(self):
        chain = EmbeddingChain(config=_config("deepinfra", target_dims=4))
        v = chain._postprocess([3.0, 4.0, 0.0, 0.0, 99.0, 99.0, 99.0, 99.0])
        # First 4 → [3, 4, 0, 0]; norm 5 → [0.6, 0.8, 0, 0].
        assert np.allclose(np.linalg.norm(v), 1.0)
        assert np.allclose(v, [0.6, 0.8, 0.0, 0.0])

    def test_normalize_disabled_keeps_raw_values(self):
        cfg = _config("deepinfra", target_dims=4)
        cfg.normalize_after_truncate = False
        chain = EmbeddingChain(config=cfg)
        v = chain._postprocess([3.0, 4.0, 0.0, 0.0, 99.0, 99.0, 99.0, 99.0])
        assert np.allclose(v, [3.0, 4.0, 0.0, 0.0])

    def test_zero_vector_handled(self):
        chain = EmbeddingChain(config=_config("deepinfra", target_dims=4))
        v = chain._postprocess([0.0] * 8)
        # Norm is 0; the `if n > 0.0` branch leaves the vector untouched.
        assert np.allclose(v, [0.0, 0.0, 0.0, 0.0])

    def test_dimension_mismatch_raises(self):
        chain = EmbeddingChain(config=_config("deepinfra"))
        # Config says native_dims=8; pass 9 floats.
        with pytest.raises(ValueError, match="expected 8 dims, got 9"):
            chain._postprocess([1.0] * 9)


class TestEmbed:
    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self):
        async with EmbeddingChain(config=_config("deepinfra")) as chain:
            assert await chain.embed([]) == []

    @pytest.mark.asyncio
    async def test_first_provider_succeeds(self, monkeypatch):
        async def fake_call(self, p, texts):
            return [[1.0] * 8 for _ in texts]

        monkeypatch.setattr(EmbeddingChain, "_call", fake_call)
        async with EmbeddingChain(
            config=_config("deepinfra", "together")
        ) as chain:
            vecs = await chain.embed(["a", "b"])
        assert len(vecs) == 2
        assert all(v.shape == (4,) for v in vecs)

    @pytest.mark.asyncio
    async def test_failover_to_next_provider(self, monkeypatch):
        """Provider 1 raises → provider 2 wins. Tracks which provider
        actually serviced the call.
        """
        attempts: list[str] = []

        async def fake_call(self, p, texts):
            attempts.append(p.provider)
            if p.provider == "deepinfra":
                raise RuntimeError("simulated 503 from deepinfra")
            return [[2.0] * 8 for _ in texts]

        monkeypatch.setattr(EmbeddingChain, "_call", fake_call)
        async with EmbeddingChain(
            config=_config("deepinfra", "together")
        ) as chain:
            vecs = await chain.embed(["a"])
        assert attempts == ["deepinfra", "together"]
        assert len(vecs) == 1

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises_aggregated(self, monkeypatch):
        async def fake_call(self, p, texts):
            raise RuntimeError(f"{p.provider} unreachable")

        monkeypatch.setattr(EmbeddingChain, "_call", fake_call)
        async with EmbeddingChain(
            config=_config("deepinfra", "together", "openrouter")
        ) as chain:
            with pytest.raises(AllProvidersFailed) as exc_info:
                await chain.embed(["a"])
        msg = str(exc_info.value)
        # Aggregated error mentions every provider.
        assert "deepinfra" in msg
        assert "together" in msg
        assert "openrouter" in msg


class TestCallEnvVar:
    @pytest.mark.asyncio
    async def test_missing_api_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("FAKE_DEEPINFRA_KEY", raising=False)
        cfg = _config("deepinfra")
        async with EmbeddingChain(config=cfg) as chain:
            # Call _call directly — bypasses the failover wrapper so we see
            # the raw RuntimeError instead of AllProvidersFailed.
            with pytest.raises(
                RuntimeError, match="missing env var FAKE_DEEPINFRA_KEY"
            ):
                await chain._call(cfg.chain[0], ["x"])


class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_aenter_creates_session_aexit_closes_it(self):
        chain = EmbeddingChain(config=_config("deepinfra"))
        assert chain._session is None
        async with chain:
            assert chain._session is not None
            assert not chain._session.closed
        # After __aexit__: _session reset to None.
        assert chain._session is None
