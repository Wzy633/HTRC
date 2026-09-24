#!/usr/bin/env python3
"""
TensorCache 全自动评估: 推理 + 评估 + 对比

用法:
  CUDA_VISIBLE_DEVICES=3 python run_tensor_eval.py

流程:
  1. TensorCache 批量推理 (180 张 eval_input, 跳过已完成)
  2. 评估 TensorCache 结果 (CD + F-Score)
  3. 评估 F3C/SSC 结果 (已有, 做对比基准)
  4. 打印对比表

输出:
  toys4k_data/eval_output_tensor/   — TensorCache 生成的 glb
  toys4k_data/eval_output_tensor_metrics.json  — 评估结果
"""

import sys, os, gc, math, json, time, argparse
from pathlib import Path
import numpy as np
from glob import glob
from collections import defaultdict

os.environ['SPCONV_ALGO'] = 'native'
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from tensors.offline import (
    configure_offline_environment,
    patch_torch_hub_offline,
    resolve_local_model_path,
)

configure_offline_environment()

# ============================================================
# Config
# ============================================================
_DATA_ROOT = os.environ.get(
    'TENSORCACHE_DATA_ROOT', os.path.join(PROJECT_ROOT, 'toys4k_data'))
_EVAL_INPUT = os.path.join(_DATA_ROOT, 'eval_input')
_GROUND_TRUTH = os.path.join(_DATA_ROOT, 'ground_truth')
_OUT_TENSOR = os.environ.get(
    'TENSORCACHE_EVAL_OUTPUT', os.path.join(PROJECT_ROOT, 'outputs', 'eval_output_tensor'))
_OUT_F3C = os.path.join(_DATA_ROOT, 'eval_output')

# F3C 论文参数
class F3CArgs:
    use_tensor_cache = True
    use_tensor_cache = True
    use_low_rank_cleanup = True
    euler_steps = 25
    effective_steps = 25
    full_sampling_ratio = 0.2
    full_sampling_end_ratio = 0.75
    anchor_ratio = 0.2
    assumed_slope = -0.07
    aggressive_cache_ratio = 0.7
    final_phase_correction_freq = 3
    full_sampling_steps = math.floor(25 * 0.2)
    full_sampling_end_steps = math.ceil(25 * 0.75)
    anchor_step = max(1, math.floor(25 * 0.2))
    seed = 42
    resolution = 16


# ============================================================
# Step 1: TensorCache 批量推理
# ============================================================
def run_inference():
    from PIL import Image
    from tqdm import tqdm
    from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils
    from tensors.pipeline import inject_tensor_cache
    import torch

    args = F3CArgs()
    patch_torch_hub_offline()
    local_model = resolve_local_model_path()
    os.makedirs(_OUT_TENSOR, exist_ok=True)

    eval_items = sorted(os.listdir(_EVAL_INPUT))
    total = len(eval_items)
    already_done = len(glob(f'{_OUT_TENSOR}/*.glb'))
    print(f'\n{"="*60}')
    print(f'Step 1: TensorCache 批量推理')
    print(f'  总数: {total}, 已完成: {already_done}, 待推理: {total - already_done}')
    print(f'{"="*60}\n')

    if already_done >= total:
        print('全部完成，跳过推理。')
        return

    print('加载 pipeline...')
    pipeline = TrellisImageTo3DPipeline.from_pretrained(local_model)
    pipeline.cuda()
    pipeline = inject_tensor_cache(pipeline, args)

    sampler_params = {
        "steps": args.effective_steps,
        "cfg_strength": 7.5,
        "decoder": pipeline.models['sparse_structure_decoder'],
        "args": args
    }

    success, fail, skipped = 0, 0, 0
    failed_list = []

    pbar = tqdm(eval_items, desc='TensorCache')
    for item_name in pbar:
        out_path = os.path.join(_OUT_TENSOR, f'{item_name}.glb')
        if os.path.exists(out_path):
            skipped += 1
            pbar.set_postfix({'done': success, 'skip': skipped, 'fail': fail})
            continue

        img_path = os.path.join(_EVAL_INPUT, item_name, '0001.png')
        try:
            image = Image.open(img_path).convert('RGB')
            outputs = pipeline.run(image, seed=args.seed, sparse_structure_sampler_params=sampler_params)

            if 'gaussian' in outputs and 'mesh' in outputs:
                glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][0], outputs['mesh'][0],
                    simplify=0.95, texture_size=1024)
                glb.export(out_path)
                success += 1
            else:
                fail += 1
                failed_list.append(f'{item_name}: missing mesh/gaussian')
        except Exception as e:
            fail += 1
            failed_list.append(f'{item_name}: {e}')

        gc.collect()
        torch.cuda.empty_cache()
        pbar.set_postfix({'done': success, 'skip': skipped, 'fail': fail})

    print(f'\n推理完成: {success} 新增, {skipped} 跳过, {fail} 失败')
    if failed_list:
        print('失败列表:')
        for f in failed_list[:10]:
            print(f'  {f}')
        if len(failed_list) > 10:
            print(f'  ... 共 {len(failed_list)} 条')


# ============================================================
# Step 2: 几何评估 (CD + F-Score)
# ============================================================
def evaluate_mesh_dir(mesh_dir, label):
    """评估一个目录下的所有 glb 文件, 返回 per-object 指标。"""
    import trimesh
    from scipy.spatial import cKDTree as KDTree

    mesh_paths = sorted(glob(os.path.join(mesh_dir, '*.glb')))
    print(f'\n评估 {label}: {len(mesh_paths)} 个 glb')

    def icp_align(source, target, max_iter=50):
        try:
            matrix, transformed, cost = trimesh.registration.icp(
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

    def find_gt(fname):
        # fname: e.g. "donut_donut_010"
        parts = fname.split('_')
        for n in range(len(parts) - 1, 1, -1):
            candidate = '_'.join(parts[-n:])
            for ext in ['mesh.obj', 'mesh.ply']:
                path = os.path.join(_GROUND_TRUTH, candidate, ext)
                if os.path.exists(path):
                    return candidate, path
        return None, None

    results = {}
    success, fail = 0, 0
    for mp in mesh_paths:
        fname = os.path.basename(mp).replace('.glb', '')
        obj_id, gt_path = find_gt(fname)
        if not gt_path:
            fail += 1
            continue
        try:
            gen = trimesh.load(mp, force='mesh')
            gt = trimesh.load(gt_path, force='mesh')
            if len(gen.vertices) == 0 or len(gt.vertices) == 0:
                fail += 1
                continue
            for m in [gen, gt]:
                b = m.bounds
                if b is None: continue
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
        except Exception:
            fail += 1

    cd_vals = [r['cd'] for r in results.values()]
    fs_vals = [r['fscore'] for r in results.values()]
    print(f'  {label}: {success} ok, {fail} fail | '
          f'CD={np.mean(cd_vals):.6f}, F-Score={np.mean(fs_vals)*100:.2f} '
          f'(n={len(cd_vals)})')
    return results


# ============================================================
# Step 3: 汇总对比
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip_inference', action='store_true', help='跳过推理, 仅评估')
    p.add_argument('--gpu', type=int, default=3)
    args_p = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args_p.gpu)

    t0 = time.time()

    # Step 1: 推理
    if not args_p.skip_inference:
        run_inference()

    # Step 2: 评估 TensorCache
    tensor_results = evaluate_mesh_dir(_OUT_TENSOR, 'TensorCache')

    # Step 3: 评估 F3C/SSC (基线)
    f3c_results = evaluate_mesh_dir(_OUT_F3C, 'F3C/SSC')

    # Step 4: 对比 (只取两个方法都成功的物体)
    common = set(tensor_results.keys()) & set(f3c_results.keys())
    t_cd = np.mean([tensor_results[k]['cd'] for k in common])
    t_fs = np.mean([tensor_results[k]['fscore'] for k in common]) * 100
    f_cd = np.mean([f3c_results[k]['cd'] for k in common])
    f_fs = np.mean([f3c_results[k]['fscore'] for k in common]) * 100

    cd_delta = (t_cd - f_cd) / f_cd * 100  # 正值 = 变差
    fs_delta = (t_fs - f_fs)  # 负值 = 变差

    print(f'\n{"="*60}')
    print(f'最终对比 (共同 {len(common)} 个物体)')
    print(f'{"="*60}')
    print(f'{"指标":<25} {"F3C/SSC":>12} {"TensorCache":>12} {"Δ":>12}')
    print(f'{"─"*25} {"─"*12} {"─"*12} {"─"*12}')
    print(f'{"Chamfer Distance":<25} {f_cd:>12.6f} {t_cd:>12.6f} {cd_delta:>+11.2f}%')
    print(f'{"F-Score (×100)":<25} {f_fs:>12.4f} {t_fs:>12.4f} {fs_delta:>+11.2f}')
    print(f'{""}  {"":>12} {"":>12} {"(负值=TensorCache更好)":>12}')
    print(f'\n论文参考: F3C τ=8: CD=0.0703, F-Score=53.75')
    print(f'           Vanilla:   CD=0.0686, F-Score=54.82')

    # 保存
    summary = {
        'tensor_cache': {
            'n': len(tensor_results),
            'cd_mean': float(np.mean([r['cd'] for r in tensor_results.values()])),
            'fscore_mean': float(np.mean([r['fscore'] for r in tensor_results.values()])),
        },
        'f3c_ssc': {
            'n': len(f3c_results),
            'cd_mean': float(np.mean([r['cd'] for r in f3c_results.values()])),
            'fscore_mean': float(np.mean([r['fscore'] for r in f3c_results.values()])),
        },
        'comparison': {
            'n_common': len(common),
            'cd_delta_pct': float(cd_delta),
            'fs_delta': float(fs_delta),
            'tensor_cd': float(t_cd),
            'f3c_cd': float(f_cd),
            'tensor_fs': float(t_fs),
            'f3c_fs': float(f_fs),
        },
        'elapsed_min': (time.time() - t0) / 60,
    }
    out_json = f'{_OUT_TENSOR}_metrics.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n结果保存到 {out_json}')
    print(f'总耗时: {summary["elapsed_min"]:.1f} 分钟')


if __name__ == '__main__':
    main()
