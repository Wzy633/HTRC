"""Cache scheduling for spatial Tucker fiber completion.

The scheduler deliberately does not claim a Candes--Recht recovery guarantee:
that theorem concerns dispersed matrix entries, whereas TensorCache observes
complete channel fibers at selected 3-D locations.  The default budget is a
transparent degrees-of-freedom (DoF) heuristic with an explicit oversampling
factor; experiments may instead request a fixed cache ratio.
"""

import math
from typing import Sequence, Tuple


class CacheLeader:
    def __init__(self):
        self.num_steps = 0
        self.resolution = 16
        self.channels = 8
        self.full_sampling_steps = 6
        self.full_sampling_end_steps = 19
        self.anchor_step = 5
        self.final_phase_correction_freq = 3

        self.budget_mode = "dof"
        self.target_cache_ratio = 0.5
        self.oversampling = 12.0
        self.min_observation_ratio = 0.2
        self.max_cache_ratio = 0.75
        self.tucker_ranks: Tuple[int, int, int] = (4, 4, 4)
        self.current_step = 0

    def set_parameters(self, args):
        self.num_steps = int(args.effective_steps)
        self.resolution = int(args.resolution)
        self.channels = int(getattr(args, "tensor_cache_channels", 8))
        self.full_sampling_steps = int(args.full_sampling_steps)
        self.full_sampling_end_steps = int(args.full_sampling_end_steps)
        self.anchor_step = int(args.anchor_step)
        self.final_phase_correction_freq = int(args.final_phase_correction_freq)
        self.budget_mode = str(getattr(args, "tensor_cache_budget_mode", "dof"))
        self.target_cache_ratio = float(
            getattr(args, "tensor_cache_target_ratio", 0.5))
        self.oversampling = float(getattr(args, "tensor_cache_oversampling", 12.0))
        self.min_observation_ratio = float(
            getattr(args, "tensor_cache_min_observation_ratio", 0.2))
        self.max_cache_ratio = float(
            getattr(args, "tensor_cache_max_ratio", 0.75))
        fixed_rank = int(getattr(args, "tensor_cache_rank", 4))
        self.tucker_ranks = (fixed_rank, fixed_rank, fixed_rank)
        self.current_step = 0

    def increase_step(self):
        self.current_step += 1

    def set_tucker_rank(self, rank):
        if isinstance(rank, Sequence) and not isinstance(rank, (str, bytes)):
            if len(rank) != 3:
                raise ValueError("expected three spatial Tucker ranks")
            self.tucker_ranks = tuple(max(1, int(item)) for item in rank)
        else:
            scalar = max(1, int(rank))
            self.tucker_ranks = (scalar, scalar, scalar)

    @property
    def tucker_rank(self):
        """Compatibility view used by older diagnostics."""
        return max(self.tucker_ranks)

    def _dof_observations(self) -> int:
        n = self.resolution
        rx, ry, rz = self.tucker_ranks
        # Spatial Tucker with an uncompressed channel mode.
        degrees_of_freedom = n * (rx + ry + rz) + rx * ry * rz * self.channels
        fibers = math.ceil(self.oversampling * degrees_of_freedom / self.channels)
        minimum = math.ceil(self.min_observation_ratio * n ** 3)
        return max(minimum, fibers)

    def get_skip_budget_for_current_step(self, current_t: float) -> int:
        del current_t
        total_tokens = self.resolution ** 3
        # The anchor must be a full observation; otherwise its estimated rank
        # is contaminated by the very completion policy it is meant to set.
        if self.current_step < self.full_sampling_steps or self.current_step <= self.anchor_step:
            return 0

        if (
            self.current_step >= self.full_sampling_end_steps
            and self.final_phase_correction_freq > 0
        ):
            offset = self.current_step - self.full_sampling_end_steps
            if (offset + 1) % self.final_phase_correction_freq == 0:
                return 0

        if self.budget_mode == "ratio":
            cache_ratio = self.target_cache_ratio
        elif self.budget_mode == "dof":
            must_observe = self._dof_observations()
            cache_ratio = 1.0 - must_observe / total_tokens
        else:
            raise ValueError(f"unknown tensor_cache_budget_mode={self.budget_mode!r}")

        cache_ratio = max(0.0, min(cache_ratio, self.max_cache_ratio))
        return max(0, min(total_tokens - 1, int(total_tokens * cache_ratio)))


LEADER = CacheLeader()
