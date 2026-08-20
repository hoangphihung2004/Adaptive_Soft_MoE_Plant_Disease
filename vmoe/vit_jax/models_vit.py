"""Fallback models_vit module for VMoE."""
from typing import Any, Optional
import flax.linen as nn
import jax.numpy as jnp


class IdentityLayer(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x


class AddPositionEmbs(nn.Module):
    posemb_init: Any = None

    @nn.compact
    def __call__(self, inputs):
        assert inputs.ndim == 3
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
        return inputs + pe


class MlpBlock(nn.Module):
    mlp_dim: int
    dtype: Any = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, *, deterministic=True):
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(features=self.mlp_dim, dtype=self.dtype, name='fc1')(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(features=actual_out_dim, dtype=self.dtype, name='fc2')(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x


class VisionTransformer(nn.Module):
    num_classes: int
    patches: Any = None
    transformer: Any = None
    hidden_size: int = 768
    representation_size: Optional[int] = None
    classifier: str = 'token'

    @nn.compact
    def __call__(self, inputs, *, train=True):
        raise NotImplementedError("Use VMoE VisionTransformerMoe model from vmoe.nn.vit_moe instead.")
