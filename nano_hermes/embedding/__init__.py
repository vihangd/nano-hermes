"""Embedding chain: bge-m3 with DeepInfra → Together → OpenRouter failover."""
from .chain import EmbeddingChain, AllProvidersFailed

__all__ = ["EmbeddingChain", "AllProvidersFailed"]
