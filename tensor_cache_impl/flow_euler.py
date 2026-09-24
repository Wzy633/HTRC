"""Deprecated compatibility import; canonical implementation is ``tensors``."""

from tensors.sampler import F3cTensorCacheSampler, TensorCacheSampler

__all__ = ["TensorCacheSampler", "F3cTensorCacheSampler"]
