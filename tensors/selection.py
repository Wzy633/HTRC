"""Temporal-risk token selection and observation-constrained Tucker completion.

The cache operates on a velocity tensor with shape ``(B, C, D, H, W)``.
Skipping one spatial token removes an entire channel fiber, so ordinary
entry-wise matrix-completion guarantees do not apply.  This module therefore
uses the previous velocity only as a temporal prior and enforces the newly
computed token fibers as hard observations during Tucker hard-imputation.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _as_tokens(value: torch.Tensor) -> torch.Tensor:
    """Return ``value`` as ``(B, N, C)`` without changing token order."""
    if value.dim() == 5:
        return value.flatten(2).transpose(1, 2).contiguous()
    if value.dim() == 3:
        return value
    raise ValueError(f"expected a 3-D token matrix or 5-D grid, got {value.shape}")


def _energy_rank(singular_values: torch.Tensor, threshold: float) -> int:
    energy = singular_values.square()
    total = energy.sum()
    if not torch.isfinite(total) or total <= 0:
        return 1
    cumulative = energy.cumsum(0) / total
    return int(torch.searchsorted(cumulative, threshold).item() + 1)


def estimate_spatial_tucker_rank(
    value: torch.Tensor,
    threshold: float = 0.95,
    max_rank: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Estimate ranks of the three spatial unfoldings.

    Channels remain uncompressed: the scientific hypothesis concerns spatial
    redundancy, not a principal/residual split along feature dimensions.
    """
    if value.dim() != 5:
        raise ValueError("spatial Tucker rank requires a (B,C,D,H,W) tensor")
    if value.shape[0] != 1:
        raise ValueError("rank estimation currently expects batch size one")

    spatial = value[0].float().permute(1, 2, 3, 0).contiguous()  # D,H,W,C
    ranks = []
    for mode in range(3):
        unfolding = spatial.movedim(mode, 0).reshape(spatial.shape[mode], -1)
        singular_values = torch.linalg.svdvals(unfolding)
        rank = _energy_rank(singular_values, threshold)
        if max_rank is not None:
            rank = min(rank, max_rank)
        ranks.append(max(1, min(rank, spatial.shape[mode])))
    return tuple(ranks)


def _tucker_project_spatial(
    tensor: torch.Tensor,
    ranks: Sequence[int],
) -> torch.Tensor:
    """Project a ``(D,H,W,C)`` tensor onto spatial Tucker ranks."""
    if tensor.dim() != 4 or len(ranks) != 3:
        raise ValueError("expected tensor (D,H,W,C) and three spatial ranks")

    factors = []
    for mode, requested_rank in enumerate(ranks):
        unfolding = tensor.movedim(mode, 0).reshape(tensor.shape[mode], -1)
        rank = max(1, min(int(requested_rank), unfolding.shape[0]))
        u, _, _ = torch.linalg.svd(unfolding, full_matrices=False)
        factors.append(u[:, :rank])

    ux, uy, uz = factors
    core = torch.einsum("ia,jb,kc,ijkd->abcd", ux, uy, uz, tensor)
    return torch.einsum("ia,jb,kc,abcd->ijkd", ux, uy, uz, core)


@dataclass
class CompletionDiagnostics:
    ranks: Tuple[int, int, int]
    iterations: int
    observed_tokens: int
    relative_update: float
    observed_residual: float


class TuckerFiberCompleter:
    """Complete the temporal velocity residual with hard data consistency.

    The stale cache is the zero-order predictor.  We therefore reconstruct
    ``delta = current - prior`` rather than projecting the full velocity field.
    Missing residual fibers start at zero, while observed residuals are clamped
    after every Tucker projection.  The final estimate is ``prior + delta``.
    """

    def __init__(
        self,
        resolution: int = 16,
        ranks: Sequence[int] = (4, 4, 4),
        iterations: int = 4,
        relaxation: float = 1.0,
    ):
        self.resolution = int(resolution)
        self.ranks = tuple(int(rank) for rank in ranks)
        self.iterations = max(1, int(iterations))
        self.relaxation = float(relaxation)
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("relaxation must be in (0, 1]")

    @torch.no_grad()
    def complete(
        self,
        prior_tokens: torch.Tensor,
        observed_indices: torch.Tensor,
        observed_values: torch.Tensor,
        ranks: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, CompletionDiagnostics]:
        prior = _as_tokens(prior_tokens).float()
        observed = _as_tokens(observed_values).float()
        if prior.shape[0] != observed.shape[0] or prior.shape[2] != observed.shape[2]:
            raise ValueError("prior and observed values have incompatible batch/channels")

        batch, token_count, channels = prior.shape
        if token_count != self.resolution ** 3:
            raise ValueError(
                f"expected {self.resolution ** 3} tokens, got {token_count}")
        if observed.shape[1] != observed_indices.numel():
            raise ValueError("observed_values must align with observed_indices")

        active_ranks = tuple(int(rank) for rank in (ranks or self.ranks))
        observed_delta = observed - prior[:, observed_indices, :]
        delta_estimate = torch.zeros_like(prior)
        delta_estimate[:, observed_indices, :] = observed_delta

        for _ in range(self.iterations):
            projected_batches = []
            for batch_index in range(batch):
                grid = delta_estimate[batch_index].reshape(
                    self.resolution, self.resolution, self.resolution, channels)
                projected_batches.append(_tucker_project_spatial(grid, active_ranks))
            projected = torch.stack(projected_batches).reshape(batch, token_count, channels)
            delta_estimate = (
                self.relaxation * projected
                + (1.0 - self.relaxation) * delta_estimate
            )
            delta_estimate[:, observed_indices, :] = observed_delta

        estimate = prior + delta_estimate
        estimate[:, observed_indices, :] = observed

        observed_error = (
            estimate[:, observed_indices, :] - observed
        ).norm() / observed.norm().clamp_min(1e-12)
        relative_update = delta_estimate.norm() / prior.norm().clamp_min(1e-12)
        diagnostics = CompletionDiagnostics(
            ranks=active_ranks,
            iterations=self.iterations,
            observed_tokens=int(observed_indices.numel()),
            relative_update=float(relative_update),
            observed_residual=float(observed_error),
        )
        return estimate.to(prior_tokens.dtype), diagnostics


class SpatialFiberSampler:
    """Select fresh token fibers from deployable historical risk estimates.

    ``residual_norm`` is the acceleration-only baseline also studied by
    Fast3DCache.  ``residual_risk`` is the proposed partial-observation-aware
    variant: scores are updated only where a fresh velocity was observed,
    carried forward elsewhere, and augmented with trend and 3-D neighbourhood
    risk.  Hierarchical staleness exempts high-confidence stable tokens from
    the ordinary streak limit while forcing low-confidence tokens to refresh.
    """

    def __init__(
        self,
        num_tokens: int = 4096,
        resolution: int = 16,
        rank_threshold: float = 0.95,
        max_staleness: int = 2,
        probe_fraction: float = 0.05,
    ):
        self.num_tokens = int(num_tokens)
        self.resolution = int(resolution)
        self.rank_threshold = float(rank_threshold)
        self.max_staleness = max(1, int(max_staleness))
        self.probe_fraction = max(0.0, min(float(probe_fraction), 1.0))
        self.strategy = "residual_risk"
        self.ssc_acceleration_weight = 0.7
        self.residual_ema_decay = 0.7
        self.residual_trend_weight = 0.5
        self.residual_spatial_weight = 0.25
        self.high_confidence_fraction = 0.25
        self.low_confidence_fraction = 0.20
        self.device = torch.device("cpu")
        self.cached_streak_counter: Optional[torch.Tensor] = None
        self.last_active_mask: Optional[torch.Tensor] = None
        self.residual_memory: Optional[torch.Tensor] = None
        self.residual_ema: Optional[torch.Tensor] = None
        self.previous_measured_residual: Optional[torch.Tensor] = None
        self.diag_log = []
        self.diag_enabled = False

    def reset(self, device="cpu", latent_channels=8):
        del latent_channels
        self.device = torch.device(device)
        self.cached_streak_counter = torch.zeros(
            self.num_tokens, device=self.device, dtype=torch.long)
        self.last_active_mask = None
        self.residual_memory = None
        self.residual_ema = None
        self.previous_measured_residual = None

    def set_hyperparameters(self, args):
        self.rank_threshold = float(
            getattr(args, "tensor_cache_rank_threshold", self.rank_threshold))
        self.max_staleness = max(
            1, int(getattr(args, "tensor_cache_max_staleness", self.max_staleness)))
        self.probe_fraction = max(
            0.0,
            min(float(getattr(args, "tensor_cache_probe_fraction", self.probe_fraction)), 1.0),
        )
        self.strategy = str(getattr(
            args, "tensor_cache_selection_strategy", self.strategy))
        self.ssc_acceleration_weight = float(getattr(
            args, "tensor_cache_ssc_acceleration_weight", self.ssc_acceleration_weight))
        self.residual_ema_decay = float(getattr(
            args, "tensor_cache_residual_ema_decay", self.residual_ema_decay))
        self.residual_trend_weight = float(getattr(
            args, "tensor_cache_residual_trend_weight", self.residual_trend_weight))
        self.residual_spatial_weight = float(getattr(
            args, "tensor_cache_residual_spatial_weight", self.residual_spatial_weight))
        self.high_confidence_fraction = float(getattr(
            args, "tensor_cache_high_confidence_fraction", self.high_confidence_fraction))
        self.low_confidence_fraction = float(getattr(
            args, "tensor_cache_low_confidence_fraction", self.low_confidence_fraction))

    @staticmethod
    def _minmax(score: torch.Tensor) -> torch.Tensor:
        score = score.float()
        return (score - score.min()) / (score.max() - score.min()).clamp_min(1e-12)

    def _update_residual_state(self, previous_residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = _as_tokens(previous_residual).float()[0].to(self.device).norm(dim=1)
        if self.residual_memory is None:
            self.residual_memory = raw.clone()
            self.residual_ema = raw.clone()
            self.previous_measured_residual = raw.clone()
            return raw, torch.zeros_like(raw)

        observed = self.last_active_mask
        if observed is None:
            observed = torch.ones_like(raw, dtype=torch.bool)
        old = self.residual_memory.clone()
        measured = torch.where(observed, raw, old)
        trend = torch.where(observed, (raw - old).clamp_min(0), torch.zeros_like(raw))
        self.residual_memory = measured
        self.residual_ema = torch.where(
            observed,
            self.residual_ema_decay * self.residual_ema
            + (1.0 - self.residual_ema_decay) * raw,
            self.residual_ema,
        )
        self.previous_measured_residual = measured
        return measured, trend

    def _selection_score(
        self,
        pred_v: torch.Tensor,
        previous_residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        tokens = _as_tokens(pred_v).float()[0].to(self.device)
        rank = 0
        trend = torch.zeros(tokens.shape[0], device=self.device)
        residual = torch.zeros(tokens.shape[0], device=self.device)
        if previous_residual is not None:
            residual, trend = self._update_residual_state(previous_residual)
        elif self.residual_ema is None:
            self.residual_memory = residual.clone()
            self.residual_ema = residual.clone()
            self.previous_measured_residual = residual.clone()

        if self.strategy == "random":
            score = torch.rand(tokens.shape[0], device=self.device)
        elif self.strategy == "leverage":
            u, singular_values, _ = torch.linalg.svd(tokens, full_matrices=False)
            rank = _energy_rank(singular_values, self.rank_threshold)
            score = u[:, :rank].square().sum(1)
        elif self.strategy == "residual_norm":
            score = residual
        elif self.strategy == "fast3d_ssc":
            velocity = tokens.norm(dim=1)
            weight = min(1.0, max(0.0, self.ssc_acceleration_weight))
            score = weight * self._minmax(residual) + (1.0 - weight) * self._minmax(velocity)
        elif self.strategy in ("residual_risk", "residual_risk_flat", "hybrid"):
            grid = self.residual_ema.view(1, 1, self.resolution, self.resolution, self.resolution)
            neighbourhood = F.max_pool3d(grid, kernel_size=3, stride=1, padding=1).flatten()
            score = (
                self.residual_ema
                + self.residual_trend_weight * trend
                + self.residual_spatial_weight * neighbourhood
            )
            if self.strategy == "hybrid":
                u, singular_values, _ = torch.linalg.svd(tokens, full_matrices=False)
                rank = _energy_rank(singular_values, self.rank_threshold)
                leverage = u[:, :rank].square().sum(1)
                score = self._minmax(score) + self._minmax(leverage)
        else:
            raise ValueError(f"unknown tensor cache selection strategy {self.strategy!r}")
        return score, rank, trend

    @torch.no_grad()
    def update_and_select(
        self,
        pred_v,
        num_to_skip: int,
        t: float,
        previous_residual: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del kwargs
        if self.cached_streak_counter is None:
            self.reset(pred_v.device)

        tokens = _as_tokens(pred_v).float()[0].to(self.device)
        token_count = tokens.shape[0]
        num_to_skip = max(0, min(int(num_to_skip), token_count - 1))
        num_active = token_count - num_to_skip

        if num_to_skip <= 0:
            self.cached_streak_counter.zero_()
            active = torch.arange(token_count, device=self.device)
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, active

        score, rank, trend = self._selection_score(pred_v, previous_residual)

        # A small deterministic spatial probe set prevents top-leverage-only
        # sampling from leaving entire regions unobserved.
        probe_count = min(num_active, round(num_active * self.probe_fraction))
        if probe_count > 0:
            probe_indices = torch.linspace(
                0, token_count - 1, steps=probe_count, device=self.device
            ).round().long().unique()
        else:
            probe_indices = torch.empty(0, device=self.device, dtype=torch.long)
        selected_mask = torch.zeros(token_count, dtype=torch.bool, device=self.device)
        selected_mask[probe_indices] = True

        remaining = num_active - int(selected_mask.sum())
        if remaining > 0:
            scores = score.masked_fill(selected_mask, float("-inf"))
            selected_mask[torch.topk(scores, k=remaining).indices] = True

        # Hierarchical staleness.  The lowest-risk group is exempt from the
        # ordinary streak threshold; its exemption is revoked automatically if
        # retained risk rises relative to the current population.  The highest-
        # risk group is always refreshed.
        if self.strategy in ("residual_risk", "hybrid"):
            stable_cut = torch.quantile(
                score, min(1.0, max(0.0, self.high_confidence_fraction)))
            unstable_cut = torch.quantile(
                score, 1.0 - min(1.0, max(0.0, self.low_confidence_fraction)))
            high_confidence = score <= stable_cut
            low_confidence = score >= unstable_cut
        else:
            high_confidence = torch.zeros_like(score, dtype=torch.bool)
            low_confidence = torch.zeros_like(score, dtype=torch.bool)
        force_stale = (
            self.cached_streak_counter >= self.max_staleness
        ) & ~high_confidence
        selected_mask |= force_stale | low_confidence
        active_indices = torch.where(selected_mask)[0]
        cached_indices = torch.where(~selected_mask)[0]

        self.cached_streak_counter[active_indices] = 0
        self.cached_streak_counter[cached_indices] += 1
        self.last_active_mask = selected_mask.clone()

        if self.diag_enabled:
            self.diag_log.append({
                "t": float(t),
                "num_to_skip": int(num_to_skip),
                "cached": int(cached_indices.numel()),
                "active": int(active_indices.numel()),
                "strategy": self.strategy,
                "r_eff": int(rank),
                "probe_count": int(probe_indices.numel()),
                "score_max": float(score.max()),
                "score_mean": float(score.mean()),
                "trend_mean": float(trend.mean()),
                "high_confidence": int(high_confidence.sum()),
                "low_confidence": int(low_confidence.sum()),
                "forced_stale_refresh": int(force_stale.sum()),
            })
        return cached_indices, active_indices


# Backward-compatible name for existing pipeline imports.
LeverageScoreTracker = SpatialFiberSampler


def estimate_leverage_scores(v_grid: torch.Tensor, rank_threshold: float = 0.95):
    tokens = _as_tokens(v_grid).float()[0]
    u, singular_values, _ = torch.linalg.svd(tokens, full_matrices=False)
    rank = _energy_rank(singular_values, rank_threshold)
    leverage = u[:, :rank].square().sum(1)
    return leverage, rank, singular_values
