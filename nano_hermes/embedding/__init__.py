"""Embedding chain: bge-m3 with DeepInfra → Together → OpenRouter failover."""
from .chain import EmbeddingChain, AllProvidersFailed, embed_cache_key

__all__ = ["EmbeddingChain", "AllProvidersFailed", "embed_cache_key"]
