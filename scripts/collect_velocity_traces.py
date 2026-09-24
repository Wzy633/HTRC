#!/usr/bin/env python3
"""Collect full-compute velocity traces for multiple local images, offline."""

import argparse
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tensors.offline import configure_offline_environment

configure_offline_environment()

from PIL import Image
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline

from tensors.offline import patch_torch_hub_offline, resolve_local_model_path
from tensors.pipeline import inject_tensor_cache


def _images(input_dir):
    preferred = sorted(input_dir.glob("*/0001.png"))
    if preferred:
        return preferred
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    return sorted({path for pattern in extensions for path in input_dir.rglob(pattern)})


def _tensor_cache_args(steps, seed):
    return SimpleNamespace(
        use_tensor_cache=True,
        use_low_rank_cleanup=True,
        effective_steps=steps,
        full_sampling_steps=math.floor(steps * 0.2),
        full_sampling_end_steps=math.ceil(steps * 0.75),
        anchor_step=max(1, math.floor(steps * 0.2)),
        final_phase_correction_freq=3,
        tensor_cache_budget_mode="ratio",
        tensor_cache_target_ratio=0.0,
        tensor_cache_max_ratio=0.0,
        tensor_cache_rank=4,
        tensor_cache_observation_mode="subset",
        tensor_cache_rank_threshold=0.95,
        tensor_cache_completion_iters=4,
        tensor_cache_completion_relaxation=1.0,
        tensor_cache_max_staleness=2,
        tensor_cache_probe_fraction=0.05,
        tensor_cache_dump_velocity=None,
        seed=seed,
        resolution=16,
    )


def main():
    parser = argparse.ArgumentParser()
    default_data = Path(os.environ.get(
        "TENSORCACHE_DATA_ROOT", PROJECT_ROOT / "toys4k_data")) / "eval_input"
    parser.add_argument("--input_dir", type=Path, default=default_data)
    parser.add_argument("--output_dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "velocity_traces")
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--max_objects", type=int, default=5)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args_cli = parser.parse_args()

    images = _images(args_cli.input_dir)
    if args_cli.max_objects is not None:
        images = images[:args_cli.max_objects]
    if not images:
        raise FileNotFoundError(f"no local images found under {args_cli.input_dir}")

    patch_torch_hub_offline()
    model_path = resolve_local_model_path(args_cli.model_path)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
    pipeline.cuda()
    cache_args = _tensor_cache_args(args_cli.steps, args_cli.seed)
    pipeline = inject_tensor_cache(pipeline, cache_args)
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(images, start=1):
        object_id = image_path.parent.name if image_path.name == "0001.png" else image_path.stem
        trace_path = args_cli.output_dir / f"{object_id}.pt"
        if trace_path.exists():
            print(f"[{index}/{len(images)}] skip existing {trace_path}")
            continue
        cache_args.tensor_cache_dump_velocity = str(trace_path)
        print(f"[{index}/{len(images)}] {image_path} -> {trace_path}")
        pipeline.run(
            Image.open(image_path).convert("RGB"),
            seed=cache_args.seed,
            sparse_structure_sampler_params={
                "steps": cache_args.effective_steps,
                "cfg_strength": 7.5,
                "decoder": pipeline.models["sparse_structure_decoder"],
                "args": cache_args,
            },
        )

    print(f"velocity traces ready under {args_cli.output_dir}")


if __name__ == "__main__":
    main()
