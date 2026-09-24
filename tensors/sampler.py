# TensorCache/sampler.py
#
# TensorCache: temporal residual-risk token selection + explicit cache budget.

from typing import *
import os, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from trellis.modules.spatial import unpatchify, patchify
from tensors.leader import LEADER
from tensors.selection import (
    SpatialFiberSampler,
    TuckerFiberCompleter,
    estimate_spatial_tucker_rank,
)


class TensorCacheSampler(FlowEulerGuidanceIntervalSampler):
    """
    Temporal-residual-guided spatial token caching sampler.

    - Token selection: historical residual risk under partial observation.
    - Cache budget: explicit ratio (recommended) or legacy Tucker DoF mode.
    - Hierarchical staleness for high/medium/low-confidence tokens.
    - Tucker completion is retained only as an opt-in negative ablation.
    """

    diag_log = []
    diag_enabled = False
    run_log = []

    def __init__(self, use_low_rank_cleanup=True, **kwargs):
        super().__init__(**kwargs)
        # ``use_low_rank_cleanup`` is retained for old entry points.  In the
        # refactored method it controls observation-constrained completion.
        self.stability_tracker = SpatialFiberSampler(
            num_tokens=LEADER.resolution ** 3,
            resolution=LEADER.resolution)
        self.completer = TuckerFiberCompleter(resolution=LEADER.resolution)
        self.model_dtype = torch.float16
        self.use_low_rank_cleanup = use_low_rank_cleanup
        self.last_completion_diagnostics = None

        if os.environ.get('TENSOR_DIAG') == '1':
            TensorCacheSampler.diag_enabled = True
            TensorCacheSampler.diag_log = []
            self.stability_tracker.diag_enabled = True

    def _init_tensor_cache_state(self, x_t, args, model):
        LEADER.set_parameters(args)
        if hasattr(model, 'dtype'):
            self.model_dtype = model.dtype
        elif hasattr(model, 'parameters'):
            try:
                self.model_dtype = next(model.parameters()).dtype
            except StopIteration:
                pass
        self.stability_tracker.set_hyperparameters(args)
        self.stability_tracker.reset(device=x_t.device, latent_channels=x_t.shape[1])
        rank = int(getattr(args, 'tensor_cache_rank', 4))
        self.completer = TuckerFiberCompleter(
            resolution=LEADER.resolution,
            ranks=(rank, rank, rank),
            iterations=int(getattr(args, 'tensor_cache_completion_iters', 4)),
            relaxation=float(getattr(args, 'tensor_cache_completion_relaxation', 1.0)),
        )

    def _run_model_core(self, tokens, t_tensor, cond, model):
        target_dtype = self.model_dtype
        original_dtype = tokens.dtype
        t_emb = model.t_embedder(t_tensor)
        if hasattr(model, 'share_mod') and model.share_mod and hasattr(model, 'adaLN_modulation'):
            t_emb = model.adaLN_modulation(t_emb)
        t_emb = t_emb.to(target_dtype)
        h = tokens.to(target_dtype)
        cond = cond.to(target_dtype)
        for block in model.blocks:
            h = block(h, t_emb, cond)
        h = h.to(original_dtype)
        h = F.layer_norm(h, h.shape[-1:])
        return model.out_layer(h)

    @torch.no_grad()
    def _estimate_rank(self, residual_grid: torch.Tensor, var_threshold=0.95):
        """Estimate spatial ranks of a fully observed temporal residual."""
        ranks = estimate_spatial_tucker_rank(
            residual_grid.float(),
            threshold=var_threshold,
            max_rank=getattr(self, 'max_tucker_rank', None),
        )
        LEADER.set_tucker_rank(ranks)
        self.completer.ranks = ranks

    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps, cfg_strength, decoder, args,
               verbose=True, cfg_interval=(0.5, 1.0), **kwargs):
        if noise.is_cuda:
            torch.cuda.synchronize(noise.device)
        sampler_started = time.perf_counter()
        self._init_tensor_cache_state(noise, args, model)
        B, C_in, D, H, W = noise.shape
        total_tokens = (D // model.patch_size) ** 3
        out_patched_channels = C_in
        if hasattr(model, 'out_channels') and hasattr(model, 'patch_size') and model.patch_size > 0:
            out_patched_channels = model.out_channels * model.patch_size ** 3

        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        rescale_t_val = 3.0
        if rescale_t_val != 1.0:
            t_seq = rescale_t_val * t_seq / (1 + (rescale_t_val - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))

        ret = edict({"samples": None, "pred_x_0_latents": []})
        last_pred_v_grid = None
        last_velocity_residual = None
        cached_v_tokens = torch.zeros(B, total_tokens, out_patched_channels,
                                       device=noise.device, dtype=torch.float32)
        velocity_trace = [] if getattr(args, 'tensor_cache_dump_velocity', None) else None

        for t, t_prev in tqdm(t_pairs, desc="TensorCache", disable=not verbose):
            current_step = LEADER.current_step
            prior_v_grid = last_pred_v_grid

            # ---- caching decision ----
            is_cache_active = False
            cached_indices, fast_update_indices = None, None
            use_tensor_cache = getattr(
                args, 'use_tensor_cache', getattr(args, 'use_f3c', False))
            if use_tensor_cache and last_pred_v_grid is not None and current_step >= LEADER.full_sampling_steps:
                num_to_skip = LEADER.get_skip_budget_for_current_step(t)
                if num_to_skip > 0 and num_to_skip < total_tokens:
                    is_cache_active = True
                    cached_indices, fast_update_indices = self.stability_tracker.update_and_select(
                        last_pred_v_grid,
                        num_to_skip,
                        t,
                        previous_residual=last_velocity_residual,
                    )

            # ---- forward pass ----
            h = patchify(sample.float(), model.patch_size)
            h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
            input_tokens_full = model.input_layer(h) + model.pos_emb[None]
            observation_mode = getattr(args, 'tensor_cache_observation_mode', 'subset')
            if is_cache_active and observation_mode == 'subset':
                input_tokens = input_tokens_full[:, fast_update_indices, :]
            elif observation_mode in ('subset', 'oracle_full'):
                input_tokens = input_tokens_full
            else:
                raise ValueError(
                    f"unknown tensor_cache_observation_mode={observation_mode!r}")

            t_tensor = torch.tensor([1000 * t] * sample.shape[0], device=sample.device)
            cond_ = cond if cond.shape[0] == B else cond.repeat(B, *([1] * (cond.ndim - 1)))

            if cfg_interval[0] <= t <= cfg_interval[1]:
                neg_cond_ = neg_cond if neg_cond.shape[0] == B else neg_cond.repeat(B, *([1] * (neg_cond.ndim - 1)))
                cond_pred_v = self._run_model_core(input_tokens, t_tensor, cond_, model)
                uncond_pred_v = self._run_model_core(input_tokens, t_tensor, neg_cond_, model)
                pred_v_tokens = uncond_pred_v + cfg_strength * (cond_pred_v - uncond_pred_v)
            else:
                pred_v_tokens = self._run_model_core(input_tokens, t_tensor, cond_, model)

            # ---- assemble velocity ----
            if is_cache_active:
                observed_v_tokens = (
                    pred_v_tokens[:, fast_update_indices, :]
                    if observation_mode == 'oracle_full'
                    else pred_v_tokens
                )
                if self.use_low_rank_cleanup:
                    final_v_tokens, completion_diag = self.completer.complete(
                        cached_v_tokens,
                        fast_update_indices,
                        observed_v_tokens,
                        ranks=LEADER.tucker_ranks,
                    )
                    self.last_completion_diagnostics = completion_diag
                else:
                    final_v_tokens = cached_v_tokens.clone()
                    final_v_tokens[:, fast_update_indices, :] = observed_v_tokens
            else:
                final_v_tokens = pred_v_tokens

            grid_size = D // model.patch_size
            expected_shape = (B, out_patched_channels, grid_size, grid_size, grid_size)
            reshaped = final_v_tokens.permute(0, 2, 1).view(*expected_shape)
            current_v_grid = unpatchify(reshaped, model.patch_size).contiguous()
            if velocity_trace is not None:
                velocity_trace.append(current_v_grid.detach().float().cpu())

            sample = sample - (t - t_prev) * current_v_grid.to(sample.dtype)

            latent_x0, _ = self._v_to_xstart_eps(x_t=sample.float(), t=t_prev, v=current_v_grid.float())
            ret.pred_x_0_latents.append(latent_x0)

            # ---- diagnostics (optional) ----
            if TensorCacheSampler.diag_enabled and is_cache_active:
                if len(self.stability_tracker.diag_log) > 0:
                    TensorCacheSampler.diag_log.append({
                        'step': int(current_step),
                        't': float(t),
                        **self.stability_tracker.diag_log[-1],
                        'tucker_ranks': list(LEADER.tucker_ranks),
                        'completion_relative_update': (
                            self.last_completion_diagnostics.relative_update
                            if self.last_completion_diagnostics is not None else None
                        ),
                        'completion_observed_residual': (
                            self.last_completion_diagnostics.observed_residual
                            if self.last_completion_diagnostics is not None else None
                        ),
                    })

            # ---- anchor step: estimate Tucker rank for the optional completion ablation ----
            if (self.use_low_rank_cleanup and use_tensor_cache
                    and current_step == LEADER.anchor_step and prior_v_grid is not None):
                self._estimate_rank(current_v_grid.float() - prior_v_grid.float())

            last_velocity_residual = (
                current_v_grid.float() - prior_v_grid.float()
                if prior_v_grid is not None else None
            )
            last_pred_v_grid = current_v_grid.float()
            cached_v_tokens = final_v_tokens

            LEADER.increase_step()

        if TensorCacheSampler.diag_enabled and TensorCacheSampler.diag_log:
            os.makedirs('outputs', exist_ok=True)
            with open('outputs/tensor_diag_log.json', 'w') as f:
                json.dump(TensorCacheSampler.diag_log, f)
        if velocity_trace is not None:
            trace_path = getattr(args, 'tensor_cache_dump_velocity')
            trace_dir = os.path.dirname(trace_path)
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
            torch.save({'velocity': torch.stack(velocity_trace)}, trace_path)

        if noise.is_cuda:
            torch.cuda.synchronize(noise.device)
        TensorCacheSampler.run_log.append({
            'selection_strategy': getattr(
                args, 'tensor_cache_selection_strategy', 'residual_risk'),
            'target_cache_ratio': float(getattr(
                args, 'tensor_cache_target_ratio', 0.0)),
            'sampler_seconds': time.perf_counter() - sampler_started,
        })

        ret.samples = sample
        return ret


# Compatibility alias for historical checkpoints/scripts. New code must use
# TensorCacheSampler and --use_tensor_cache.
F3cTensorCacheSampler = TensorCacheSampler
