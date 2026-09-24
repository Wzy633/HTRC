# TensorCache/pipeline.py

def inject_tensor_cache(pipeline, args):
    """Replace Trellis sparse structure sampler with TensorCache sampler."""
    original_sampler = pipeline.sparse_structure_sampler

    from tensors.sampler import TensorCacheSampler
    use_lr = getattr(args, 'use_low_rank_cleanup', True)
    cache_sampler = TensorCacheSampler(
        sigma_min=original_sampler.sigma_min,
        use_low_rank_cleanup=use_lr)

    pipeline.sparse_structure_sampler = cache_sampler
    print(f"GS Stage: {type(original_sampler).__name__} -> "
          f"{type(cache_sampler).__name__} "
          f"(TensorCache: {getattr(args, 'tensor_cache_selection_strategy', 'residual_risk')} "
          f"+ hierarchical staleness).")
    return pipeline
