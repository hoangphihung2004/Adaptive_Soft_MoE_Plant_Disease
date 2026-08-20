"""Fallback models_mixer module for VMoE."""
from typing import Any, Optional
import flax.linen as nn


class MlpMixer(nn.Module):
    num_classes: int
    patches: Any = None
    num_blocks: int = 8
    hidden_dim: int = 512
    tokens_mlp_dim: int = 256
    channels_mlp_dim: int = 2048

    @nn.compact
    def __call__(self, inputs, *, train=True):
        raise NotImplementedError("MlpMixer is not used in VMoE potato classification.")
