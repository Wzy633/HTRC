#!/usr/bin/env python3
"""Offline multi-object falsification test for residual TensorCache completion.

The input may be one ``.pt`` trace or a directory containing traces. Each trace
stores ``velocity`` with shape ``(T,C,D,H,W)`` or ``(T,1,C,D,H,W)``. Object,
not timestep, is the independent unit in the emitted aggregate table.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tensors.selection import (  # noqa: E402
    TuckerFiberCompleter,
    estimate_leverage_scores,
    estimate_spatial_tucker_rank,
)


def _tokens(grid):
    return grid.flatten(2).transpose(1, 2).contiguous()


def _trace_files(path):
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.pt"))
        if files:
            return files
    raise FileNotFoundError(f"no .pt velocity traces found at {path}")


def _load_sequence(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    sequence = payload.get("velocity") if isinstance(payload, dict) else payload
    if sequence is None:
        raise ValueError(f"{path} has no 'velocity' tensor")
    if sequence.dim() == 6 and sequence.shape[1] == 1:
        sequence = sequence[:, 0]
    if sequence.dim() != 5 or sequence.shape[0] < 2:
        raise ValueError(f"{path}: expected (T,C,D,H,W), T>=2; got {sequence.shape}")
    return sequence.float()


def _unit_rank(values):
    """Map scores to [0,1] ranks without scale assumptions."""
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.linspace(0.0, 1.0, values.numel())
    return ranks


def _select_active(previous, previous_residual, active_count, strategy, generator):
    token_count = previous.shape[-3] * previous.shape[-2] * previous.shape[-1]
    if strategy == "random":
        return torch.randperm(token_count, generator=generator)[:active_count]
    if strategy == "spatial_uniform":
        return torch.linspace(0, token_count - 1, steps=active_count).round().long().unique()
    residual_score = _tokens(previous_residual)[0].norm(dim=1)
    if strategy == "residual_norm":
        return torch.topk(residual_score, k=active_count).indices
    if strategy == "leverage":
        leverage, _, _ = estimate_leverage_scores(previous)
        return torch.topk(leverage.cpu(), k=active_count).indices
    if strategy == "hybrid":
        leverage, _, _ = estimate_leverage_scores(previous)
        score = _unit_rank(leverage.cpu()) + _unit_rank(residual_score.cpu())
        return torch.topk(score, k=active_count).indices
    raise ValueError(f"unknown strategy {strategy!r}")


def relative_error(prediction, target):
    return float((prediction - target).norm() / target.norm().clamp_min(1e-12))


def _ranks_for_sequence(sequence, mode, threshold, anchor_step, fixed_rank):
    if mode == "fixed":
        return (fixed_rank, fixed_rank, fixed_rank)
    if mode == "anchor":
        step = max(1, min(anchor_step, sequence.shape[0] - 1))
        residual = (sequence[step] - sequence[step - 1]).unsqueeze(0)
        return estimate_spatial_tucker_rank(residual, threshold=threshold)
    return None


def _aggregate_by_object(records):
    per_object = defaultdict(list)
    for record in records:
        key = (record["object"], record["cache_ratio"], record["strategy"])
        per_object[key].append(record)

    object_rows = []
    for (object_id, cache_ratio, strategy), rows in per_object.items():
        object_rows.append({
            "object": object_id,
            "cache_ratio": cache_ratio,
            "strategy": strategy,
            "steps": len(rows),
            "stale_mean": statistics.fmean(row["stale_relative_error"] for row in rows),
            "completion_mean": statistics.fmean(
                row["completion_relative_error"] for row in rows),
            "win_rate": statistics.fmean(
                row["completion_relative_error"] < row["stale_relative_error"]
                for row in rows),
        })

    grouped = defaultdict(list)
    for row in object_rows:
        grouped[(row["cache_ratio"], row["strategy"])].append(row)
    aggregate = []
    for (cache_ratio, strategy), rows in sorted(grouped.items()):
        improvements = [
            (row["stale_mean"] - row["completion_mean"]) / max(row["stale_mean"], 1e-12)
            for row in rows
        ]
        aggregate.append({
            "cache_ratio": cache_ratio,
            "strategy": strategy,
            "n_objects": len(rows),
            "object_win_rate": statistics.fmean(value > 0 for value in improvements),
            "mean_relative_improvement": statistics.fmean(improvements),
            "median_relative_improvement": statistics.median(improvements),
        })
    return object_rows, aggregate


def _aggregate_selection(object_rows):
    by_key = defaultdict(list)
    for row in object_rows:
        by_key[(row["cache_ratio"], row["strategy"])].append(row)
    random_by_object = {
        (row["object"], row["cache_ratio"]): row["stale_mean"]
        for row in object_rows if row["strategy"] == "random"
    }
    output = []
    for (cache_ratio, strategy), rows in sorted(by_key.items()):
        gains = []
        for row in rows:
            baseline = random_by_object.get((row["object"], cache_ratio))
            if baseline is not None:
                gains.append((baseline - row["stale_mean"]) / max(baseline, 1e-12))
        output.append({
            "cache_ratio": cache_ratio,
            "strategy": strategy,
            "n_objects": len(rows),
            "stale_mean": statistics.fmean(row["stale_mean"] for row in rows),
            "object_win_vs_random": statistics.fmean(gain > 0 for gain in gains) if gains else None,
            "mean_improvement_vs_random": statistics.fmean(gains) if gains else None,
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("velocity_path", type=Path)
    parser.add_argument("--cache_ratios", default="0.2,0.35,0.5,0.65")
    parser.add_argument(
        "--strategies",
        default="random,spatial_uniform,leverage,residual_norm,hybrid")
    parser.add_argument("--selection_only", action="store_true",
                        help="Skip Tucker completion and evaluate stale selection only.")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--rank_threshold", type=float, default=0.95)
    parser.add_argument("--rank_mode", choices=("anchor", "per_step", "fixed"),
                        default="anchor")
    parser.add_argument("--anchor_step", type=int, default=5)
    parser.add_argument("--fixed_rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/oracle_completion.json"))
    args = parser.parse_args()

    cache_ratios = [float(value) for value in args.cache_ratios.split(",")]
    strategies = [value.strip() for value in args.strategies.split(",")]
    records = []

    for object_index, trace_file in enumerate(_trace_files(args.velocity_path)):
        sequence = _load_sequence(trace_file)
        object_id = trace_file.stem
        generator = torch.Generator().manual_seed(args.seed + object_index)
        anchor_ranks = _ranks_for_sequence(
            sequence, args.rank_mode, args.rank_threshold,
            args.anchor_step, args.fixed_rank)

        # Step 1 has no preceding residual and is excluded equally for every
        # strategy so temporal selectors remain comparable.
        for step in range(2, sequence.shape[0]):
            previous = sequence[step - 1].unsqueeze(0)
            current = sequence[step].unsqueeze(0)
            residual = current - previous
            previous_residual = (
                sequence[step - 1] - sequence[step - 2]).unsqueeze(0)
            ranks = anchor_ranks or estimate_spatial_tucker_rank(
                residual, threshold=args.rank_threshold)
            prior_tokens, current_tokens = _tokens(previous), _tokens(current)
            completer = TuckerFiberCompleter(
                resolution=previous.shape[-1], ranks=ranks,
                iterations=args.iterations)

            for cache_ratio in cache_ratios:
                active_count = max(
                    1, round((1.0 - cache_ratio) * prior_tokens.shape[1]))
                for strategy in strategies:
                    active = _select_active(
                        previous, previous_residual, active_count, strategy, generator)
                    if args.selection_only:
                        completed = prior_tokens.clone()
                        completed[:, active, :] = current_tokens[:, active, :]
                        observed_residual = 0.0
                    else:
                        completed, diagnostics = completer.complete(
                            prior_tokens, active, current_tokens[:, active, :], ranks=ranks)
                        observed_residual = diagnostics.observed_residual
                    missing = torch.ones(prior_tokens.shape[1], dtype=torch.bool)
                    missing[active] = False
                    records.append({
                        "object": object_id,
                        "trace": str(trace_file),
                        "step": step,
                        "cache_ratio": cache_ratio,
                        "strategy": strategy,
                        "ranks": list(ranks),
                        "stale_relative_error": relative_error(
                            prior_tokens[:, missing], current_tokens[:, missing]),
                        "completion_relative_error": relative_error(
                            completed[:, missing], current_tokens[:, missing]),
                        "all_token_relative_error": relative_error(
                            completed, current_tokens),
                        "observed_residual": observed_residual,
                    })

    object_rows, aggregate = _aggregate_by_object(records)
    selection_aggregate = _aggregate_selection(object_rows)
    report = {
        "source": str(args.velocity_path),
        "rank_mode": args.rank_mode,
        "n_objects": len({record["object"] for record in records}),
        "object_summary": object_rows,
        "aggregate": aggregate,
        "selection_aggregate": selection_aggregate,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {len(records)} records from {report['n_objects']} objects to {args.output}")
    for row in aggregate:
        print(
            f"cache={row['cache_ratio']:.0%} strategy={row['strategy']} "
            f"n={row['n_objects']} object-win={row['object_win_rate']:.1%} "
            f"mean-improvement={row['mean_relative_improvement']:+.1%}")
    if args.selection_only:
        print("selection-only stale error (relative to random):")
        for row in selection_aggregate:
            gain = row["mean_improvement_vs_random"]
            gain_text = "baseline" if gain is None else f"{gain:+.1%}"
            print(
                f"cache={row['cache_ratio']:.0%} strategy={row['strategy']:15s} "
                f"n={row['n_objects']} stale={row['stale_mean']:.6f} "
                f"vs-random={gain_text}")


if __name__ == "__main__":
    main()
