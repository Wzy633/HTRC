# HTRC: Observation-Aware Temporal Residual Caching for Sparse 3D Flow Sampling

HTRC is a training-free caching method for accelerating the sparse-structure stage of image-conditioned 3D generation with [TRELLIS](https://github.com/microsoft/TRELLIS). It reuses token velocities while tracking the temporal evidence available to justify reuse.

**When a token is skipped, its current change is unobserved—not measured as zero.** HTRC retains residual memories at skipped locations, combines temporal history with local 3D neighborhood information, and assigns reuse horizons according to token confidence.

[Overview](#overview) · [Setup](#setup) · [Quick Start](#quick-start) · [Evaluation](#evaluation) · [Code Structure](#code-structure)

## Overview

- **Observation-aware memory:** update residual evidence only where fresh network outputs are available.
- **Risk-based refresh:** prioritize tokens using retained temporal changes and neighborhood information.
- **Hierarchical reuse:** allow stable tokens to remain cached longer while refreshing uncertain tokens more frequently.
- **No additional training:** use the pretrained TRELLIS model and retain each skipped token's own cached velocity.

The manuscript reports a **15.48% reduction in sparse-structure stage latency** on 179 Toys4K objects using one NVIDIA RTX 3090, with small changes in the evaluated geometry metrics. This is a stage-level result, not an end-to-end generation speedup. Higher cache rates do not necessarily produce lower latency.

## Setup

### 1. Prepare the TRELLIS environment

Follow the upstream [TRELLIS installation instructions](https://github.com/microsoft/TRELLIS#-installation) to prepare its CUDA/PyTorch environment and compiled dependencies. Use the environment that can run TRELLIS image-to-3D inference and mesh export. The shell examples below use Bash.

This checkout does not include a root installation script or a pinned dependency manifest. The `dataset_toolkits/` directory contains data preprocessing tools; its `setup.sh` prepares the dependencies for those tools.

After activating the environment, change to this project's `HTRC` directory. Run all commands below from that directory so Python imports the bundled `trellis/` implementation.

### 2. Prepare pretrained models

Download the [TRELLIS-image-large checkpoint](https://huggingface.co/microsoft/TRELLIS-image-large) in advance, preserving its directory structure. Set the path to the local snapshot containing `pipeline.json`:

```bash
export TENSORCACHE_MODEL_PATH=/absolute/path/to/TRELLIS-image-large
```

The inference entry point loads local checkpoints and disables network fallback for TRELLIS model discovery and Torch Hub. Also prepare the DINOv2 Torch Hub repository and checkpoint used by the model's `image_cond_model` setting. Background removal requires the `rembg` U2Net weights when the input does not already contain a usable alpha channel. Running upstream TRELLIS inference once while online is a practical way to populate these caches; use the same environment and cache locations for HTRC.

You can alternatively pass `--model_path /absolute/path/to/TRELLIS-image-large` to the single-image command. The resolver also recognizes `TRELLIS_MODEL_PATH` and existing Hugging Face snapshots.

## Quick Start

### Generate a 3D asset with HTRC

```bash
python -m tensor_cache_impl.example \
  --use_tensor_cache \
  --model_path "$TENSORCACHE_MODEL_PATH" \
  --image_path assets/example_image/typical_misc_monster_chest.png \
  --tensor_cache_budget_mode ratio \
  --tensor_cache_target_ratio 0.50 \
  --tensor_cache_selection_strategy residual_risk \
  --euler_steps 25 \
  --seed 42 \
  --output_dir outputs \
  --output_name htrc
```

The generated textured asset is saved to `outputs/htrc.glb`.

Specify `--tensor_cache_budget_mode ratio` explicitly: the parser defaults to the experimental `dof` budget mode. Setting the target ratio alone does not select a fixed-ratio budget. Tucker completion is disabled by default and is not part of the HTRC configuration above.

### Generate an uncached reference

```bash
python -m tensor_cache_impl.example \
  --model_path "$TENSORCACHE_MODEL_PATH" \
  --image_path assets/example_image/typical_misc_monster_chest.png \
  --euler_steps 25 \
  --seed 42 \
  --output_dir outputs \
  --output_name vanilla
```

Use the same image, seed, model, and sampling steps when comparing outputs.

### Main options

| Option | Default | Meaning |
| --- | --- | --- |
| `--use_tensor_cache` | Off | Enable caching in the sparse-structure sampler. |
| `--tensor_cache_budget_mode` | `dof` | Use `ratio` for an explicit target cache fraction. |
| `--tensor_cache_target_ratio` | `0.50` | Target cache fraction in ratio mode; realized reuse also depends on refresh constraints and the step schedule. |
| `--tensor_cache_selection_strategy` | `residual_risk` | HTRC token-selection policy. |
| `--tensor_cache_max_staleness` | `2` | Cache-age control, adjusted by the confidence policy. |
| `--tensor_cache_high_confidence_fraction` | `0.25` | Low-risk fraction exempt from the streak limit. |
| `--tensor_cache_low_confidence_fraction` | `0.20` | High-risk fraction forced to refresh. |
| `--euler_steps` | `25` | Number of sparse-structure sampling steps. |
| `--seed` | `42` | Generation seed. |

Additional settings are documented in [`tensors/argparser.py`](tensors/argparser.py). In a configured environment, run `python -m tensor_cache_impl.example --help` for the full CLI.

## Evaluation

[`scripts/ablation.py`](scripts/ablation.py) runs selection-policy and cache-ratio experiments with explicit configurations. Prepare rendered input images and the ground-truth assets required by the evaluation code. Input images are read from `<input_dir>/<object_id>/0001.png`.

For example, compare uncached inference with HTRC over a cache-budget sweep:

```bash
python scripts/ablation.py \
  --strategies vanilla,residual_risk \
  --cache_ratios 0.20,0.35,0.50,0.65 \
  --input_dir /absolute/path/to/toys4k_data/eval_input \
  --ground_truth_dir /absolute/path/to/toys4k_data/ground_truth \
  --output_root outputs/ablation
```

Set `TENSORCACHE_MODEL_PATH` before running this command. Use `--max_images 5` for a small trial. Reports are written under the output root, including `ablation_report.json`. Use a fresh output directory when changing an experiment's settings, because existing generated assets may be reused.

Available selection policies include `random`, `leverage`, `residual_norm`, `fast3d_ssc`, `residual_risk_flat`, `residual_risk`, and `hybrid`; `vanilla` disables caching. These are experimental comparison options, with `residual_risk` selecting HTRC.

The commands above are experiment entry points. Reproducing a specific manuscript table additionally requires its object split, settings, and timing definition. Sparse-structure stage time, sampler time, and whole-pipeline time cover different operations and should be reported separately. Likewise, cache fractions averaged over cache-active steps are not interchangeable with fractions measured over a fixed sampling window.

The older `scripts/eval_full.py` uses a separate configuration, including enabled low-rank cleanup; it is not the default HTRC reproduction entry point. Trace collection and completion diagnostics are provided in `scripts/collect_velocity_traces.py` and `scripts/oracle_completion.py`.

## Code Structure

```text
HTRC/
├── tensors/                  # Canonical caching implementation
│   ├── sampler.py            # Cached flow sampler
│   ├── selection.py          # Selection policies and diagnostic utilities
│   ├── leader.py             # Cache state and scheduling
│   ├── pipeline.py           # TRELLIS sampler integration
│   ├── argparser.py          # Single-image CLI options
│   └── offline.py            # Local model and cache discovery
├── tensor_cache_impl/        # Compatibility imports and single-image entry point
│   └── example.py
├── trellis/                  # Bundled TRELLIS implementation
├── scripts/                  # Evaluation, ablations, and trace diagnostics
├── evaluation/               # Geometry and performance evaluation utilities
├── dataset_toolkits/         # Data preprocessing tools
├── assets/                   # Example inputs
└── test_completion.py        # Selection/completion unit tests
```

New integrations should import from `tensors`. The `tensor_cache_impl` directory retains legacy import paths and the runnable single-image example; it is not a second implementation of HTRC.

## Troubleshooting

- **Local model snapshot not found:** point `--model_path` or `TENSORCACHE_MODEL_PATH` to the directory containing `pipeline.json`, rather than a model identifier or an individual weight file.
- **DINOv2 repository or checkpoint not cached:** populate Torch Hub's repository and checkpoint caches before starting the offline entry point. Check that the online and offline runs use the same cache location.
- **Missing CUDA extension or attention backend:** complete the upstream TRELLIS dependency installation in the active environment.
- **Increasing the cache target does not improve speed:** measure wall-clock latency; the requested cache fraction does not account for all scheduling, attention, and decoding costs.

## Acknowledgments

This project builds on [TRELLIS](https://github.com/microsoft/TRELLIS). We thank its authors and the developers of the underlying libraries. Please cite TRELLIS when using its pretrained models and follow the applicable upstream code, model, and dataset terms.
