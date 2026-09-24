import unittest

import torch

from tensors.leader import CacheLeader
from tensors.selection import (
    SpatialFiberSampler,
    TuckerFiberCompleter,
    estimate_spatial_tucker_rank,
)


class CompletionTests(unittest.TestCase):
    def test_observed_fibers_are_preserved_exactly(self):
        torch.manual_seed(0)
        prior = torch.randn(1, 64, 3)
        observed_indices = torch.tensor([0, 3, 11, 25, 63])
        observed = torch.randn(1, observed_indices.numel(), 3)
        completer = TuckerFiberCompleter(
            resolution=4, ranks=(2, 2, 2), iterations=3)
        completed, diagnostics = completer.complete(
            prior, observed_indices, observed)
        torch.testing.assert_close(completed[:, observed_indices], observed)
        self.assertEqual(diagnostics.observed_residual, 0.0)

    def test_full_observation_is_identity(self):
        torch.manual_seed(1)
        target = torch.randn(1, 64, 2)
        indices = torch.arange(64)
        completer = TuckerFiberCompleter(
            resolution=4, ranks=(1, 1, 1), iterations=2)
        completed, _ = completer.complete(target * 0, indices, target)
        torch.testing.assert_close(completed, target)

    def test_zero_temporal_residual_keeps_stale_prior(self):
        torch.manual_seed(2)
        prior = torch.randn(1, 64, 2)
        indices = torch.tensor([0, 7, 19, 42])
        completer = TuckerFiberCompleter(
            resolution=4, ranks=(2, 2, 2), iterations=3)
        completed, _ = completer.complete(
            prior, indices, prior[:, indices, :])
        torch.testing.assert_close(completed, prior)

    def test_spatial_rank_estimator_returns_three_valid_ranks(self):
        grid = torch.randn(1, 2, 4, 4, 4)
        ranks = estimate_spatial_tucker_rank(grid, threshold=0.9, max_rank=3)
        self.assertEqual(len(ranks), 3)
        self.assertTrue(all(1 <= rank <= 3 for rank in ranks))

    def test_ratio_budget_is_explicit(self):
        class Args:
            effective_steps = 10
            resolution = 4
            full_sampling_steps = 0
            full_sampling_end_steps = 10
            anchor_step = 0
            final_phase_correction_freq = 0
            tensor_cache_budget_mode = "ratio"
            tensor_cache_target_ratio = 0.5
            tensor_cache_max_ratio = 0.75
            tensor_cache_rank = 2

        leader = CacheLeader()
        leader.set_parameters(Args())
        self.assertEqual(leader.get_skip_budget_for_current_step(0.5), 32)

    def test_residual_memory_is_not_zeroed_for_cached_tokens(self):
        sampler = SpatialFiberSampler(
            num_tokens=8, resolution=2, max_staleness=2, probe_fraction=0.0)
        sampler.strategy = "residual_risk"
        sampler.high_confidence_fraction = 0.25
        sampler.low_confidence_fraction = 0.0
        sampler.reset()
        velocity = torch.ones(1, 1, 2, 2, 2)
        residual = torch.arange(1, 9, dtype=torch.float32).view(1, 1, 2, 2, 2)
        cached, _ = sampler.update_and_select(
            velocity, num_to_skip=4, t=0.5, previous_residual=residual)
        remembered = sampler.residual_memory.clone()

        sampler.update_and_select(
            velocity, num_to_skip=4, t=0.4,
            previous_residual=torch.zeros_like(residual))
        torch.testing.assert_close(
            sampler.residual_memory[cached], remembered[cached])

    def test_high_confidence_tokens_bypass_streak_refresh(self):
        sampler = SpatialFiberSampler(
            num_tokens=8, resolution=2, max_staleness=1, probe_fraction=0.0)
        sampler.strategy = "residual_risk"
        sampler.high_confidence_fraction = 0.5
        sampler.low_confidence_fraction = 0.0
        sampler.reset()
        velocity = torch.ones(1, 1, 2, 2, 2)
        residual = torch.arange(8, dtype=torch.float32).view(1, 1, 2, 2, 2)
        first_cached, _ = sampler.update_and_select(
            velocity, num_to_skip=4, t=0.5, previous_residual=residual)
        second_cached, _ = sampler.update_and_select(
            velocity, num_to_skip=4, t=0.4, previous_residual=residual)
        exempt = torch.tensor(sorted(set(first_cached.tolist()) & set(second_cached.tolist())))
        self.assertGreater(exempt.numel(), 0)
        self.assertTrue(torch.all(sampler.cached_streak_counter[exempt] > 1))


if __name__ == "__main__":
    unittest.main()
