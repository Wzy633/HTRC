# TensorCache/argparser.py

import argparse
import math


def parse_args():
    parser = argparse.ArgumentParser(
        description="Temporal-residual-guided spatial token caching for TRELLIS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    mode_group = parser.add_argument_group('Mode Control & Steps')
    mode_group.add_argument("--use_tensor_cache", action="store_true",
                            help="Enable TensorCache acceleration.")
    mode_group.add_argument("--use_f3c", action="store_true", help=argparse.SUPPRESS)
    mode_group.add_argument("--tensor_cache_completion", action="store_true",
                            help="Enable Tucker completion as an opt-in ablation.")
    mode_group.add_argument("--no_low_rank_cleanup", action="store_true",
                            help=argparse.SUPPRESS)
    mode_group.add_argument("--euler_steps", type=int, default=25,
                            help="Number of Euler sampling steps.")

    strategy_group = parser.add_argument_group('TensorCache Scheduling Strategy')
    strategy_group.add_argument("--full_sampling_ratio", type=float, default=0.2,
                                help="Phase 1 (warmup): fraction of steps with full computation.")
    strategy_group.add_argument("--full_sampling_end_ratio", type=float, default=0.75,
                                help="Phase 3 start: fraction of steps where aggressive caching begins.")
    strategy_group.add_argument("--anchor_ratio", type=float, default=0.2,
                                help="Fraction of steps at which Tucker rank is estimated.")
    strategy_group.add_argument("--final_phase_correction_freq", type=int, default=3,
                                help="Phase 3: every N steps, do full computation (correction).")
    strategy_group.add_argument("--tensor_cache_budget_mode", choices=("dof", "ratio"),
                                default="dof",
                                help="Use Tucker degrees of freedom or a fixed cache ratio.")
    strategy_group.add_argument("--tensor_cache_target_ratio", type=float, default=0.5,
                                help="Cache ratio when --tensor_cache_budget_mode=ratio.")
    strategy_group.add_argument(
        "--tensor_cache_selection_strategy",
        choices=("random", "leverage", "residual_norm", "fast3d_ssc",
                 "residual_risk_flat", "residual_risk", "hybrid"),
        default="residual_risk",
        help="Token ranking rule; fast3d_ssc is an explicit comparison baseline.")
    strategy_group.add_argument("--tensor_cache_ssc_acceleration_weight", type=float,
                                default=0.7,
                                help="Acceleration weight in the Fast3DCache SSC baseline.")
    strategy_group.add_argument("--tensor_cache_residual_ema_decay", type=float,
                                default=0.7,
                                help="EMA decay for partial-observation residual risk.")
    strategy_group.add_argument("--tensor_cache_residual_trend_weight", type=float,
                                default=0.5,
                                help="Weight on positive residual-risk trends.")
    strategy_group.add_argument("--tensor_cache_residual_spatial_weight", type=float,
                                default=0.25,
                                help="Weight on 3-D neighbourhood risk propagation.")
    strategy_group.add_argument("--tensor_cache_high_confidence_fraction", type=float,
                                default=0.25,
                                help="Lowest-risk fraction exempt from the streak limit.")
    strategy_group.add_argument("--tensor_cache_low_confidence_fraction", type=float,
                                default=0.20,
                                help="Highest-risk fraction forced to refresh.")
    strategy_group.add_argument("--tensor_cache_max_ratio", type=float, default=0.75,
                                help="Safety cap applied to every cache budget.")
    strategy_group.add_argument("--tensor_cache_min_observation_ratio", type=float,
                                default=0.2,
                                help="Minimum fresh-token fraction in DoF mode.")
    strategy_group.add_argument("--tensor_cache_oversampling", type=float, default=12.0,
                                help="Oversampling multiplier on Tucker parameter DoF.")
    strategy_group.add_argument("--tensor_cache_rank", type=int, default=4,
                                help="Initial/max spatial Tucker rank per axis.")
    strategy_group.add_argument("--tensor_cache_rank_threshold", type=float, default=0.95,
                                help="Energy threshold for adaptive spatial rank estimation.")
    strategy_group.add_argument("--tensor_cache_completion_iters", type=int, default=4,
                                help="Hard-imputation Tucker iterations per cached step.")
    strategy_group.add_argument("--tensor_cache_completion_relaxation", type=float,
                                default=1.0,
                                help="Relaxation applied after each Tucker projection.")
    strategy_group.add_argument("--tensor_cache_max_staleness", type=int, default=2,
                                help="Maximum consecutive missing observations per token.")
    strategy_group.add_argument("--tensor_cache_probe_fraction", type=float, default=0.05,
                                help="Fraction of active tokens reserved for spatial probes.")
    strategy_group.add_argument("--tensor_cache_observation_mode",
                                choices=("subset", "oracle_full"), default="subset",
                                help="oracle_full validates completion without claiming speedup.")
    strategy_group.add_argument("--tensor_cache_factor", type=float, default=None,
                                help="Deprecated and ignored; use explicit budget parameters.")

    io_group = parser.add_argument_group('I/O & Internal Options')
    io_group.add_argument("--output_dir", type=str, default="outputs")
    io_group.add_argument("--model_path", type=str, default=None,
                          help="Local TRELLIS snapshot; network lookup is disabled.")
    io_group.add_argument("--image_path", type=str,
                          default="assets/example_image/typical_creature_elephant.png")
    io_group.add_argument("--output_name", type=str, default="sample_output")
    io_group.add_argument("--seed", type=int, default=42)
    io_group.add_argument("--resolution", type=int, default=16)
    io_group.add_argument("--tensor_cache_dump_velocity", type=str, default=None,
                          help="Optional .pt path for an offline oracle velocity trace.")

    args = parser.parse_args()

    args.effective_steps = args.euler_steps
    args.use_tensor_cache = args.use_tensor_cache or args.use_f3c
    if args.use_tensor_cache:
        args.full_sampling_steps = math.floor(args.effective_steps * args.full_sampling_ratio)
        args.full_sampling_end_steps = math.ceil(args.effective_steps * args.full_sampling_end_ratio)
        calculated_anchor_step = math.floor(args.effective_steps * args.anchor_ratio)
        args.anchor_step = max(1, calculated_anchor_step)
    args.use_low_rank_cleanup = (
        args.tensor_cache_completion and not args.no_low_rank_cleanup)
    return args
