#!/usr/bin/env python3
"""End-to-end selection and cache-ratio ablation.

GPU selection is intentionally controlled by ``CUDA_VISIBLE_DEVICES`` in the
terminal, not by a project default.
"""

import sys, os, gc, math, json, time, argparse, zlib
from pathlib import Path
import numpy as np
from glob import glob
from collections import defaultdict

os.environ['SPCONV_ALGO'] = 'native'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from tensors.offline import (
    configure_offline_environment,
    patch_torch_hub_offline,
    resolve_local_model_path,
)

configure_offline_environment()
_DATA_ROOT = os.environ.get(
    'TENSORCACHE_DATA_ROOT', os.path.join(PROJECT_ROOT, 'toys4k_data'))
_EVAL_INPUT = os.path.join(_DATA_ROOT, 'eval_input')
_GROUND_TRUTH = os.path.join(_DATA_ROOT, 'ground_truth')
_BASE_OUT = os.environ.get(
    'TENSORCACHE_OUTPUT_ROOT', os.path.join(PROJECT_ROOT, 'outputs', 'ablation'))


def make_args(cache_ratio, strategy):
    """Build explicit, reproducible arguments for one cache ratio."""
    class Args:
        use_tensor_cache = strategy != 'vanilla'
        use_low_rank_cleanup = False
        euler_steps = 25
        effective_steps = 25
        full_sampling_ratio = 0.2
        full_sampling_end_ratio = 0.75
        anchor_ratio = 0.2
        final_phase_correction_freq = 3
        full_sampling_steps = math.floor(25 * 0.2)
        full_sampling_end_steps = math.ceil(25 * 0.75)
        anchor_step = max(1, math.floor(25 * 0.2))
        tensor_cache_budget_mode = 'ratio'
        tensor_cache_target_ratio = cache_ratio
        tensor_cache_max_ratio = cache_ratio
        tensor_cache_selection_strategy = strategy
        tensor_cache_ssc_acceleration_weight = 0.7
        tensor_cache_residual_ema_decay = 0.7
        tensor_cache_residual_trend_weight = 0.5
        tensor_cache_residual_spatial_weight = 0.25
        tensor_cache_high_confidence_fraction = 0.25
        tensor_cache_low_confidence_fraction = 0.20
        tensor_cache_min_observation_ratio = 0.2
        tensor_cache_oversampling = 12.0
        tensor_cache_rank = 4
        tensor_cache_rank_threshold = 0.95
        tensor_cache_completion_iters = 4
        tensor_cache_completion_relaxation = 1.0
        tensor_cache_max_staleness = 2
        tensor_cache_probe_fraction = 0.0
        tensor_cache_observation_mode = 'subset'
        seed = 42
        resolution = 16
    return Args()


def run_inference(cache_ratio, strategy, out_dir, max_images=None):
    """Run one selection strategy at one explicit target cache ratio."""
    from PIL import Image
    from tqdm import tqdm
    from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils
    from tensors.pipeline import inject_tensor_cache
    from tensors.leader import LEADER
    import torch

    args = make_args(cache_ratio, strategy)
    is_vanilla = strategy == 'vanilla'
    os.makedirs(out_dir, exist_ok=True)

    eval_items = sorted(os.listdir(_EVAL_INPUT))
    if max_images:
        eval_items = eval_items[:max_images]

    already_done = len(glob(f'{out_dir}/*.glb'))
    print(f'\nStrategy={strategy}, cache ratio={cache_ratio:.0%}: '
          f'{len(eval_items)} images, {already_done} already done')

    if already_done >= len(eval_items):
        print('  All done, skipping inference.')
        return None

    # ---- Must be set before sampler construction. ----
    os.environ['TENSOR_DIAG'] = '1'
    patch_torch_hub_offline()
    local_model = resolve_local_model_path()

    # Load pipeline (fresh to reset LEADER state)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(local_model)
    pipeline.cuda()
    if not is_vanilla:
        pipeline = inject_tensor_cache(pipeline, args)

    # 兜底: 如果 __init__ 里没设上, 这里直接强开 tracker 的 diag
    sampler = pipeline.sparse_structure_sampler
    if not is_vanilla and not sampler.stability_tracker.diag_enabled:
        sampler.stability_tracker.diag_enabled = True
        print('  (diag enabled via fallback)')

    from tensors.sampler import TensorCacheSampler
    TensorCacheSampler.diag_enabled = not is_vanilla
    TensorCacheSampler.diag_log = []
    TensorCacheSampler.run_log = []

    sampler_params = {
        "steps": args.effective_steps,
        "cfg_strength": 7.5,
    }
    if not is_vanilla:
        sampler_params.update({
            "decoder": pipeline.models['sparse_structure_decoder'],
            "args": args,
        })

    sampler_seconds = []
    original_sample = sampler.sample
    def _timed_sample(*sample_args, **sample_kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_sample(*sample_args, **sample_kwargs)
        torch.cuda.synchronize()
        sampler_seconds.append(time.perf_counter() - started)
        return result
    sampler.sample = _timed_sample

    # Also capture rank per image
    rank_log = []

    # Patch set_tucker_rank to log
    _original_set_rank = LEADER.set_tucker_rank
    def _logging_set_rank(rank):
        rank_tuple = tuple(int(value) for value in rank)
        rank_log.append({'ranks': rank_tuple, 'rank': max(rank_tuple)})
        _original_set_rank(rank)
    if not is_vanilla:
        LEADER.set_tucker_rank = _logging_set_rank

    success, fail, skipped = 0, 0, 0
    pipeline_seconds = []
    records_path = os.path.join(out_dir, 'inference_records.json')
    if os.path.exists(records_path):
        try:
            with open(records_path, 'r') as f:
                inference_records = json.load(f)
        except (OSError, json.JSONDecodeError):
            inference_records = {}
    else:
        inference_records = {}

    def _save_records():
        temporary = records_path + '.tmp'
        with open(temporary, 'w') as f:
            json.dump(inference_records, f, indent=2)
        os.replace(temporary, records_path)

    pbar = tqdm(eval_items, desc=f'{strategy} cache={cache_ratio:.0%}')
    for item_name in pbar:
        out_path = os.path.join(out_dir, f'{item_name}.glb')
        if os.path.exists(out_path):
            skipped += 1
            inference_records.setdefault(item_name, {
                'status': 'skipped_existing',
                'output_path': out_path,
                'sampler_seconds': None,
                'pipeline_seconds': None,
            })
            pbar.set_postfix({'done': success, 'skip': skipped, 'fail': fail})
            continue

        img_path = os.path.join(_EVAL_INPUT, item_name, '0001.png')
        sampler_start_index = len(sampler_seconds)
        diag_start_index = len(TensorCacheSampler.diag_log)
        pipeline_elapsed = None
        try:
            image = Image.open(img_path).convert('RGB')
            torch.cuda.synchronize()
            pipeline_started = time.perf_counter()
            outputs = pipeline.run(image, seed=args.seed,
                                   sparse_structure_sampler_params=sampler_params)
            torch.cuda.synchronize()
            pipeline_elapsed = time.perf_counter() - pipeline_started
            pipeline_seconds.append(pipeline_elapsed)

            if 'gaussian' in outputs and 'mesh' in outputs:
                glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][0], outputs['mesh'][0],
                    simplify=0.95, texture_size=1024)
                glb.export(out_path)
                success += 1
                status = 'complete'
                error_text = None
            else:
                fail += 1
                status = 'failed_missing_output'
                error_text = 'pipeline result did not contain mesh and gaussian'
        except Exception as e:
            fail += 1
            status = 'failed_exception'
            error_text = repr(e)

        object_diag = TensorCacheSampler.diag_log[diag_start_index:]
        object_phase2 = [
            row for row in object_diag
            if row['step'] >= args.full_sampling_steps
            and row['step'] < args.full_sampling_end_steps
        ]
        inference_records[item_name] = {
            'status': status,
            'error': error_text,
            'input_path': img_path,
            'output_path': out_path,
            'sampler_seconds': (
                sampler_seconds[sampler_start_index]
                if len(sampler_seconds) > sampler_start_index else None),
            'pipeline_seconds': pipeline_elapsed,
            'phase2_cache_rate': (
                float(np.mean([row['cached'] / (16 ** 3) for row in object_phase2]))
                if object_phase2 else (0.0 if is_vanilla else None)),
            'high_confidence_mean': (
                float(np.mean([row.get('high_confidence', 0) for row in object_phase2]))
                if object_phase2 else None),
            'forced_stale_refresh_mean': (
                float(np.mean([row.get('forced_stale_refresh', 0) for row in object_phase2]))
                if object_phase2 else None),
            'n_cached_steps_logged': len(object_diag),
        }
        _save_records()

        gc.collect()
        torch.cuda.empty_cache()
        pbar.set_postfix({'done': success, 'skip': skipped, 'fail': fail})

    _save_records()

    # Restore
    if not is_vanilla:
        LEADER.set_tucker_rank = _original_set_rank
    TensorCacheSampler.diag_enabled = False

    # Summarize diagnostics
    diag = TensorCacheSampler.diag_log
    TensorCacheSampler.diag_log = []
    TensorCacheSampler.run_log = []

    total_tokens = 16 ** 3
    phase2_steps = [d for d in diag
                    if d['step'] >= args.full_sampling_steps
                    and d['step'] < args.full_sampling_end_steps]
    phase3_steps = [d for d in diag
                    if d['step'] >= args.full_sampling_end_steps]

    summary = {
        'selection_strategy': strategy,
        'target_cache_ratio': cache_ratio,
        'n_images': success,
        'n_failures': fail,
        'n_skipped': skipped,
        'n_steps_total': len(diag),
        'inference_records_path': records_path,
        'n_inference_records': len(inference_records),
        'pipeline_seconds_mean': (
            float(np.mean(pipeline_seconds)) if pipeline_seconds else None),
        'sampler_seconds_mean': (
            float(np.mean(sampler_seconds)) if sampler_seconds else None),
        'avg_rank': float(np.mean([r['rank'] for r in rank_log])) if rank_log else None,
        'ranks': [r['rank'] for r in rank_log],
    }

    if is_vanilla:
        summary['phase2_cache_rate'] = 0.0
        summary['overall_cache_rate'] = 0.0

    if phase2_steps:
        summary['phase2_avg_skip_budget'] = float(np.mean([d['num_to_skip'] for d in phase2_steps]))
        summary['phase2_avg_cached'] = float(np.mean([d['cached'] for d in phase2_steps]))
        summary['phase2_cache_rate'] = float(np.mean([d['cached'] / total_tokens for d in phase2_steps]))
        summary['phase2_avg_r_eff'] = float(np.mean([d['r_eff'] for d in phase2_steps]))

    if phase3_steps:
        summary['phase3_avg_cached'] = float(np.mean([d['cached'] for d in phase3_steps]))
        summary['phase3_cache_rate'] = float(np.mean([d['cached'] / total_tokens for d in phase3_steps]))

    # Overall (all phases including phase1 with 0 cache)
    if diag:
        all_cached = [d['cached'] for d in diag]
        # assume phase1 has 0 cached
        n_phase1 = args.full_sampling_steps * success
        all_cached_padded = [0] * n_phase1 + all_cached
        summary['overall_cache_rate'] = float(np.mean([c / total_tokens for c in all_cached_padded]))

    # Save diag
    diag_path = os.path.join(out_dir, 'diag_summary.json')
    with open(diag_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Diag saved to {diag_path}')
    p2cr = summary.get('phase2_cache_rate')
    avg_r = summary.get('avg_rank')
    p2cr_str = f'{p2cr:.1%}' if p2cr is not None else 'N/A'
    avg_r_str = f'{avg_r:.1f}' if avg_r is not None else 'N/A'
    print(f'  Phase2 cache rate: {p2cr_str}  (avg rank={avg_r_str})')

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def evaluate_mesh_dir(mesh_dir, label, allowed_names=None):
    """Evaluate CD/F-Score on an optional, explicitly matched object set."""
    import trimesh
    from scipy.spatial import cKDTree as KDTree

    mesh_paths = sorted(glob(os.path.join(mesh_dir, '*.glb')))
    if allowed_names is not None:
        allowed_names = set(allowed_names)
        mesh_paths = [
            path for path in mesh_paths
            if Path(path).stem in allowed_names
        ]
    if not mesh_paths:
        print(f'  {label}: no meshes found under {mesh_dir}')
        return {}

    print(f'\nEvaluating {label}: {len(mesh_paths)} meshes')

    def icp_align(source, target, max_iter=50):
        try:
            _, transformed, _ = trimesh.registration.icp(
                source, target, max_iterations=max_iter)
            return transformed
        except Exception:
            return source

    def compute_metrics(gen_model, gt_model, n_points=10000, tau=0.05):
        if not hasattr(gen_model, 'vertices') or len(gen_model.vertices) == 0:
            return None, None
        if not hasattr(gt_model, 'vertices') or len(gt_model.vertices) == 0:
            return None, None
        gen_points, _ = trimesh.sample.sample_surface(gen_model, n_points)
        gt_points, _ = trimesh.sample.sample_surface(gt_model, n_points)
        if len(gen_points) == 0 or len(gt_points) == 0:
            return None, None
        gen_points = icp_align(gen_points, gt_points)
        gt_tree, pred_tree = KDTree(gt_points), KDTree(gen_points)
        pred_to_gt, _ = gt_tree.query(gen_points)
        gt_to_pred, _ = pred_tree.query(gt_points)
        precision = np.mean((pred_to_gt < tau).astype(float))
        recall = np.mean((gt_to_pred < tau).astype(float))
        fscore = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        cd = (np.mean(pred_to_gt) + np.mean(gt_to_pred)) / 2
        return fscore, cd

    gt_files = sorted(path for path in (
        glob(os.path.join(_GROUND_TRUTH, '**', '*.obj'), recursive=True)
        + glob(os.path.join(_GROUND_TRUTH, '**', '*.ply'), recursive=True)
    ) if os.path.isfile(path))

    def normalize_name(value):
        value = str(value).lower()
        for suffix in ('.glb', '.png', '.jpg', '.jpeg', '.webp'):
            if value.endswith(suffix):
                value = value[:-len(suffix)]
        return value.strip('_-. ')

    gt_index = []
    for path in gt_files:
        file_path = Path(path)
        keys = {normalize_name(file_path.stem)}
        if file_path.stem.lower() == 'mesh':
            keys.add(normalize_name(file_path.parent.name))
        gt_index.append((keys, path))

    def find_gt(fname):
        normalized = normalize_name(fname)
        # Fast exact paths for the standard Toys4K layout.
        exact_candidates = []
        for dirname in (fname, normalized):
            for ext in ('mesh.obj', 'mesh.ply'):
                exact_candidates.append(os.path.join(_GROUND_TRUTH, dirname, ext))
        for path in exact_candidates:
            if os.path.isfile(path):
                return path

        # Compatibility with historical output names carrying prefixes.
        parts = fname.split('_')
        for n in range(len(parts) - 1, 1, -1):
            candidate = '_'.join(parts[-n:])
            for ext in ['mesh.obj', 'mesh.ply']:
                path = os.path.join(_GROUND_TRUTH, candidate, ext)
                if os.path.isfile(path):
                    return path

        matches = []
        for keys, path in gt_index:
            for key in keys:
                if key and (normalized == key
                            or normalized.endswith(key)
                            or key.endswith(normalized)):
                    matches.append((len(key), path))
        if matches:
            matches.sort(reverse=True)
            return matches[0][1]
        return None

    results = {}
    success, fail = 0, 0
    missing_gt = []
    metric_errors = []
    for mp in mesh_paths:
        fname = os.path.basename(mp).replace('.glb', '')
        gt_path = find_gt(fname)
        if not gt_path:
            fail += 1
            if len(missing_gt) < 5:
                missing_gt.append(fname)
            continue
        try:
            # trimesh surface sampling is stochastic. Seed it per object so all
            # methods use identical metric sampling without adding experiment seeds.
            np.random.seed(zlib.crc32(fname.encode('utf-8')) & 0xffffffff)
            gen = trimesh.load(mp, force='mesh')
            gt = trimesh.load(gt_path, force='mesh')
            if len(gen.vertices) == 0 or len(gt.vertices) == 0:
                fail += 1
                continue
            for m in [gen, gt]:
                b = m.bounds
                if b is None:
                    continue
                c = (b[1] + b[0]) / 2
                m.apply_translation(-c)
                ext = np.max(m.extents)
                if ext > 0:
                    m.apply_scale(1.0 / ext)
            fscore, cd = compute_metrics(gen, gt)
            if fscore is not None:
                results[fname] = {'fscore': float(fscore), 'cd': float(cd)}
                success += 1
            else:
                fail += 1
        except Exception as error:
            fail += 1
            if len(metric_errors) < 3:
                metric_errors.append(f'{fname}: {type(error).__name__}: {error}')

    cd_vals = [r['cd'] for r in results.values()]
    fs_vals = [r['fscore'] for r in results.values()]
    if cd_vals:
        print(f'  {label}: {success} ok, {fail} fail | '
              f'CD={np.mean(cd_vals):.6f}, F-Score={np.mean(fs_vals)*100:.2f} '
              f'(n={len(cd_vals)})')
    else:
        print(f'  {label}: 0 ok, {fail} fail | mesh_dir={mesh_dir}')
    if missing_gt:
        print(f'    missing GT examples: {missing_gt}')
        print(f'    GT root: {_GROUND_TRUTH} ({len(gt_files)} obj/ply files indexed)')
    if metric_errors:
        print(f'    metric error examples: {metric_errors}')
    return results


def main():
    global _EVAL_INPUT, _GROUND_TRUTH, _BASE_OUT
    parser = argparse.ArgumentParser(description='Temporal residual-risk caching ablation')
    parser.add_argument('--cache_ratios', type=str, default='0.2,0.35,0.5,0.65',
                       help='Comma-separated explicit target cache ratios')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Limit eval images per cache ratio (None=all)')
    parser.add_argument(
        '--strategies', type=str,
        default='random,residual_norm,fast3d_ssc,residual_risk_flat,residual_risk',
        help='Comma-separated token selection strategies')
    parser.add_argument('--skip_inference', action='store_true',
                       help='Skip inference, only evaluate existing outputs')
    parser.add_argument('--input_dir', type=str, default=_EVAL_INPUT,
                        help='Local evaluation-input directory')
    parser.add_argument('--ground_truth_dir', type=str, default=_GROUND_TRUTH,
                        help='Local ground-truth directory')
    parser.add_argument('--output_root', type=str, default=_BASE_OUT,
                        help='Ablation output root containing strategy subdirectories')
    args_p = parser.parse_args()

    _EVAL_INPUT = os.path.abspath(args_p.input_dir)
    _GROUND_TRUTH = os.path.abspath(args_p.ground_truth_dir)
    _BASE_OUT = os.path.abspath(args_p.output_root)
    print(f'Input dir: {_EVAL_INPUT}')
    print(f'Ground truth dir: {_GROUND_TRUTH}')
    print(f'Output root: {_BASE_OUT}')

    cache_ratios = [float(x.strip()) for x in args_p.cache_ratios.split(',')]
    strategies = [x.strip() for x in args_p.strategies.split(',') if x.strip()]
    valid_strategies = {
        'vanilla', 'random', 'leverage', 'residual_norm', 'fast3d_ssc',
        'residual_risk_flat', 'residual_risk', 'hybrid'}
    if not strategies or any(strategy not in valid_strategies for strategy in strategies):
        parser.error(f'strategies must be selected from {sorted(valid_strategies)}')
    if any(ratio < 0 or ratio >= 1 for ratio in cache_ratios):
        parser.error('cache ratios must be in [0, 1)')
    print(f'Ablation cache ratios: {cache_ratios}')
    print(f'Ablation strategies: {strategies}')
    print(f'Max images: {args_p.max_images or "all"}')

    t0 = time.time()
    all_summaries = []

    eval_names = sorted(os.listdir(_EVAL_INPUT))
    if args_p.max_images:
        eval_names = eval_names[:args_p.max_images]

    def ratios_for(strategy):
        return [0.0] if strategy == 'vanilla' else cache_ratios

    for strategy in strategies:
        for cache_ratio in ratios_for(strategy):
            out_dir = os.path.join(
                _BASE_OUT, strategy, f'cache_{cache_ratio:.2f}')
            if not args_p.skip_inference:
                summary = run_inference(
                    cache_ratio, strategy, out_dir, args_p.max_images)
                if summary:
                    all_summaries.append(summary)

    # Evaluate all
    print(f'\n{"="*70}')
    print('Geometry Evaluation')
    print(f'{"="*70}')

    ratio_results = {}
    for strategy in strategies:
        for cache_ratio in ratios_for(strategy):
            out_dir = os.path.join(
                _BASE_OUT, strategy, f'cache_{cache_ratio:.2f}')
            label = f'{strategy} target={cache_ratio:.0%}'
            # Evaluate the files actually present in this configuration. Final
            # comparisons below use the intersection of object identifiers, so
            # filename decoration cannot accidentally remove every result.
            results = evaluate_mesh_dir(out_dir, label)
            if results:
                ratio_results[(strategy, cache_ratio)] = results

    vanilla_results = ratio_results.get(('vanilla', 0.0), {})
    if vanilla_results:
        baseline_strategy = 'vanilla'
        baseline_results = vanilla_results
    else:
        baseline_strategy = 'f3c_external'
        available_names = sorted({
            name
            for results in ratio_results.values()
            for name in results
        })
        baseline_results = evaluate_mesh_dir(
            os.path.join(_DATA_ROOT, 'eval_output'),
            'F3C/SSC matched fallback',
            allowed_names=available_names or eval_names,
        )

    # Load diag summaries for cache rate info
    diag_info = {}
    for strategy in strategies:
        for cache_ratio in ratios_for(strategy):
            diag_path = os.path.join(
                _BASE_OUT, strategy, f'cache_{cache_ratio:.2f}', 'diag_summary.json')
            if os.path.exists(diag_path):
                with open(diag_path) as f:
                    diag_info[(strategy, cache_ratio)] = json.load(f)

    # Summary table
    print(f'\n{"="*90}')
    print('Ablation Summary: Temporal Residual-Risk Caching')
    print(f'{"="*90}')
    header = f'{"Strategy":>16} {"Target CR":>10} {"Actual CR":>10} {"CD":>10} {"F-Score":>10} {"vs baseline":>12}'
    print(header)
    print('-' * 90)

    baseline_cd = (
        float(np.mean([r['cd'] for r in baseline_results.values()]))
        if baseline_results else None)

    rows = []
    for strategy in strategies:
      for cache_ratio in ratios_for(strategy):
        diag = diag_info.get((strategy, cache_ratio), {})
        results = ratio_results.get((strategy, cache_ratio), {})

        if not results:
            continue

        matched_names = sorted(set(results) & set(baseline_results))
        analysis_results = (
            {name: results[name] for name in matched_names}
            if baseline_results else results)
        if not analysis_results:
            continue
        cd_mean = np.mean([r['cd'] for r in analysis_results.values()])
        fs_mean = np.mean([r['fscore'] for r in analysis_results.values()]) * 100
        actual_cr = diag.get('phase2_cache_rate', 0)
        sampler_seconds = diag.get('sampler_seconds_mean')
        pipeline_seconds_mean = diag.get('pipeline_seconds_mean')
        cd_delta = (
            (cd_mean - np.mean([baseline_results[name]['cd'] for name in matched_names]))
            / np.mean([baseline_results[name]['cd'] for name in matched_names]) * 100
            if matched_names else None)
        fs_delta = (
            fs_mean - np.mean(
                [baseline_results[name]['fscore'] for name in matched_names]) * 100
            if matched_names else None)

        target_cr = cache_ratio

        delta_text = f'{cd_delta:+.2f}%' if cd_delta is not None else 'N/A'
        print(f'{strategy:>16} {target_cr:>10.0%} {actual_cr:>10.1%} '
              f'{cd_mean:>10.6f} {fs_mean:>10.2f} {delta_text:>12}')

        rows.append({
            'selection_strategy': strategy,
            'target_cache_rate': target_cr,
            'actual_cache_rate': actual_cr,
            'cd_mean': cd_mean,
            'fs_mean': fs_mean,
            'sampler_seconds_mean': sampler_seconds,
            'pipeline_seconds_mean': pipeline_seconds_mean,
            'cd_delta_pct': cd_delta,
            'fscore_delta_points': fs_delta,
            'n_meshes': len(analysis_results),
            'n_matched_baseline': len(matched_names),
        })

    # Find best (max cache rate with CD not degraded, i.e., CD delta <= 0 or small positive)
    if rows:
        print(f'\n{"="*70}')
        print('Analysis: CD degradation vs cache rate')
        print(f'{"="*70}')
        for r in sorted(rows, key=lambda x: (x['selection_strategy'], x['actual_cache_rate'])):
            if r['cd_delta_pct'] is None:
                print(f'  {r["selection_strategy"]:16s} '
                      f'CR={r["actual_cache_rate"]:.1%}  '
                      f'CD={r["cd_mean"]:.6f}  [NO_BASELINE]')
                continue
            status = 'OK' if r['cd_delta_pct'] < 1.0 else (
                'WARN' if r['cd_delta_pct'] < 3.0 else 'DEGRADED')
            print(f'  {r["selection_strategy"]:16s} CR={r["actual_cache_rate"]:.1%}  CD={r["cd_mean"]:.6f}  '
                  f'Δ={r["cd_delta_pct"]:+.2f}%  [{status}]')

        best = max(
            [r for r in rows
             if r['cd_delta_pct'] is not None and r['cd_delta_pct'] < 1.0],
            key=lambda x: x['actual_cache_rate'],
            default=None
        )
        if best:
            print(f'\n  Best safe configuration: {best["selection_strategy"]}, '
                  f'cache={best["actual_cache_rate"]:.1%} '
                  f'(target={best["target_cache_rate"]:.0%}, '
                  f'CD delta={best["cd_delta_pct"]:+.2f}%)')

    # Save full report
    report = {
        'cache_ratios': cache_ratios,
        'strategies': strategies,
        'baseline_strategy': baseline_strategy,
        'baseline_cd': baseline_cd,
        'per_object_results': {
            f'{strategy}@{cache_ratio:.2f}': results
            for (strategy, cache_ratio), results in ratio_results.items()
        },
        'rows': rows,
        'elapsed_min': (time.time() - t0) / 60,
    }
    report_path = os.path.join(_BASE_OUT, 'ablation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nReport saved to {report_path}')
    print(f'Total time: {report["elapsed_min"]:.1f} min')


if __name__ == '__main__':
    main()
