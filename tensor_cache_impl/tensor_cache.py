"""Deprecated compatibility imports for the canonical ``tensors`` package."""

from tensors.selection import (
    CompletionDiagnostics,
    LeverageScoreTracker,
    SpatialFiberSampler,
    TuckerFiberCompleter,
    estimate_leverage_scores,
    estimate_spatial_tucker_rank,
)

__all__ = [
    "CompletionDiagnostics",
    "LeverageScoreTracker",
    "SpatialFiberSampler",
    "TuckerFiberCompleter",
    "estimate_leverage_scores",
    "estimate_spatial_tucker_rank",
]
