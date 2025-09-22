import numpy as np

# this is needed to resolve circlualr imports
import torch


def ten(c: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy array to torch tensor on specified device.

    Args:
        c: Input numpy array.
        device: Target device for the tensor.

    Returns:
        A torch tensor on the specified device.
    """
    return torch.from_numpy(c).to(device).double()


def unwrap_state(state: dict) -> tuple[np.ndarray, np.ndarray]:
    return state["agents"].copy(), state["apples"].copy()
