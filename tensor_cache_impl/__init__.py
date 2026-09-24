"""Backward-compatible entry points for TensorCache.

New integrations should import from :mod:`tensors`; this package keeps the
historical ``python -m tensor_cache_impl.example`` command working.
"""

from tensors.pipeline import inject_tensor_cache
from tensors.sampler import TensorCacheSampler

__all__ = ["TensorCacheSampler", "inject_tensor_cache"]
