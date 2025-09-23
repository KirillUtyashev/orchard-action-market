import numpy as np

import logging
from config import LOG_DIR

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


def setup_state_logger():
    """Sets up a dedicated logger to write state/reward info to states.log."""
    # 1. Create a logger object.
    #    We give it a unique name to distinguish it from other loggers.
    state_logger = logging.getLogger("StateLogger")

    # 2. Set the logging level. DEBUG is the lowest, so it will capture everything.
    state_logger.setLevel(logging.DEBUG)

    # 3. Prevent messages from bubbling up to the root logger (avoids duplicate printing)
    state_logger.propagate = False

    # 4. Create a file handler to write to our specific file.
    #    'a' mode means it will append to the file if it exists.
    file_handler = logging.FileHandler(LOG_DIR / "states.log", mode="w")

    # 5. Create a simple formatter.
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)

    # 6. Add the handler to the logger.
    #    (Add a check to prevent adding handlers on every import)
    if not state_logger.handlers:
        state_logger.addHandler(file_handler)

    return state_logger


state_logger = setup_state_logger()
