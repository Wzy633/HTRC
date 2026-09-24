# tensor_cache_impl/example.py
#
# Single-image inference with TensorCache (leverage score + rank-based budget).
# Usage:
#   python -m tensor_cache_impl.example --use_tensor_cache --tensor_cache_target_ratio 0.5

import sys, os

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from tensors.offline import configure_offline_environment

configure_offline_environment()

from PIL import Image
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils
from tensors.argparser import parse_args
from tensors.offline import patch_torch_hub_offline, resolve_local_model_path
from tensors.pipeline import inject_tensor_cache


def main():
    args = parse_args()
    model_path = resolve_local_model_path(args.model_path)
    patch_torch_hub_offline()

    print(f"Loading Trellis pipeline offline from {model_path} ...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
    pipeline.cuda()
    print("Pipeline loaded.")

    image = Image.open(args.image_path)

    if args.use_tensor_cache:
        print("TensorCache (Leverage Score + Rank-based Budget) loading...")
        pipeline = inject_tensor_cache(pipeline, args)
        sampler_params = {
            "steps": args.effective_steps,
            "cfg_strength": 7.5,
            "decoder": pipeline.models['sparse_structure_decoder'],
            "args": args
        }
    else:
        print("Vanilla mode (no caching).")
        sampler_params = {
            "steps": args.effective_steps,
            "cfg_strength": 7.5
        }

    print(f"Generating with seed={args.seed}, steps={args.effective_steps}...")
    outputs = pipeline.run(
        image,
        seed=args.seed,
        sparse_structure_sampler_params=sampler_params
    )

    if 'gaussian' not in outputs or 'mesh' not in outputs:
        print("Error: missing gaussian or mesh in output.")
        return

    print("Post-processing → glb ...")
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.output_name}.glb")
    glb.export(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
