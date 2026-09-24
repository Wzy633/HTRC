# TensorCache inference examples (rank-based budget)

# Quick: ~50% cache rate (rank≈3)
python -m tensor_cache_impl.example --use_tensor_cache --euler_steps 25 \
    --image_path /path/to/image.png \
    --output_name test_tensor \
    --output_dir outputs \
    --tensor_cache_factor 14.0

# High cache: ~70% (rank≈3)
python -m tensor_cache_impl.example --use_tensor_cache --euler_steps 25 \
    --image_path /path/to/image.png \
    --output_name test_highcr \
    --output_dir outputs \
    --tensor_cache_factor 8.0

# Conservative: ~30% (rank≈3)
python -m tensor_cache_impl.example --use_tensor_cache --euler_steps 25 \
    --image_path /path/to/image.png \
    --output_name test_lowcr \
    --output_dir outputs \
    --tensor_cache_factor 19.6

# Vanilla (no caching)
python -m tensor_cache_impl.example --euler_steps 25 \
    --image_path /path/to/image.png \
    --output_name test_vanilla
