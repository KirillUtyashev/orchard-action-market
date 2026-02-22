"""Seed management: set once at startup, use module-level RNG everywhere."""

import random as _stdlib_random

import torch

# Module-level RNG — seeded by set_all_seeds(), used by all stochastic code.
rng: _stdlib_random.Random = _stdlib_random.Random()


def set_all_seeds(seed: int) -> None:
    """Seed all RNGs. Call once at startup."""
    global rng
    rng = _stdlib_random.Random(seed)
    torch.manual_seed(seed)
    _stdlib_random.seed(seed)
