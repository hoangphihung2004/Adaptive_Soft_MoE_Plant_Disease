"""Fallback checkpoint module for vit_jax."""

def load(prefix, **kwargs):
    raise NotImplementedError("Legacy vit_jax checkpoints are not supported in fallback mode. Use Orbax checkpoints.")

def load_pretrained(pretrained_path, **kwargs):
    raise NotImplementedError("Legacy vit_jax checkpoints are not supported in fallback mode.")
