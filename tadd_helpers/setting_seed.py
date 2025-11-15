import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """
    Sets the seed for all major libraries (random, numpy, and pytorch)
    to ensure reproducibility.
    """
    print(f"🌟 Setting all seeds to {seed}...")
    # 1. Standard Python 'random' module
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # 4. PyTorch Deterministic settings (important for CUDA)
        # Note: This can sometimes slow down training.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("✅ Seeds and deterministic settings configured.")


def set_step_seed(step: int):
    """Resets the seed for numpy and random for a single simulation step."""
    np.random.seed(step)
    random.seed(step)
