#!/usr/bin/env python3
"""Offline smoke test using the first locally deployed Toys4K image."""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from tensors.offline import configure_offline_environment

configure_offline_environment()

from PIL import Image
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

from tensors.offline import patch_torch_hub_offline, resolve_local_model_path
from tensors.pipeline import inject_tensor_cache


class Args:
    use_tensor_cache = True
    use_low_rank_cleanup = True
    effective_steps = 25
    full_sampling_steps = 5
    full_sampling_end_steps = 19
    anchor_step = 5
    final_phase_correction_freq = 3
    tensor_cache_budget_mode = "ratio"
    tensor_cache_target_ratio = 0.5
    tensor_cache_max_ratio = 0.75
    tensor_cache_rank = 4
    tensor_cache_observation_mode = "subset"
    seed = 42
    resolution = 16


def main():
    data_root = Path(os.environ.get(
        "TENSORCACHE_DATA_ROOT", PROJECT_ROOT / "toys4k_data"))
    images = sorted((data_root / "eval_input").glob("*/0001.png"))
    if not images:
        raise FileNotFoundError(
            f"no local test images under {data_root / 'eval_input'}")

    patch_torch_hub_offline()
    model_path = resolve_local_model_path()
    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
    pipeline.cuda()
    args = Args()
    pipeline = inject_tensor_cache(pipeline, args)

    outputs = pipeline.run(
        Image.open(images[0]).convert("RGB"),
        seed=args.seed,
        sparse_structure_sampler_params={
            "steps": args.effective_steps,
            "cfg_strength": 7.5,
            "decoder": pipeline.models["sparse_structure_decoder"],
            "args": args,
        },
    )
    output = PROJECT_ROOT / "test_output.glb"
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0], outputs["mesh"][0], simplify=0.95, texture_size=1024)
    glb.export(output)
    print(f"SUCCESS: {output}")


if __name__ == "__main__":
    main()
