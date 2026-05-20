"""Config validation edge cases."""
import pytest
from pydantic import ValidationError
from nano_hermes.config import EmbeddingConfig, RetrievalConfig, SkillStatsConfig


class TestConfigValidation:
    def test_target_dims_zero_raises(self):
        with pytest.raises(ValidationError):
            EmbeddingConfig(target_dims=0)

    def test_target_dims_negative_raises(self):
        with pytest.raises(ValidationError):
            EmbeddingConfig(target_dims=-1)

    def test_fts_k_zero_raises(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(fts_k=0)

    def test_vec_k_zero_raises(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(vec_k=0)

    def test_final_k_zero_raises(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(final_k=0)

    def test_promotion_threshold_zero_raises(self):
        with pytest.raises(ValidationError):
            SkillStatsConfig(promotion_threshold=0)

    def test_valid_defaults_pass(self):
        EmbeddingConfig()
        RetrievalConfig()
        SkillStatsConfig()
