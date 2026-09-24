# tensor_cache_impl/trellis_pipeline.py

from tensors.pipeline import inject_tensor_cache


def update_trellis_pipeline_for_f3c(pipeline, args):
    """Deprecated alias retained only for old external scripts."""
    return inject_tensor_cache(pipeline, args)


__all__ = ["inject_tensor_cache", "update_trellis_pipeline_for_f3c"]
