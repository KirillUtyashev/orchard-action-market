from pathlib import Path
import numpy as np
import torch


def get_config():
    side_length = 10
    num_agents = 2
    S = np.zeros((side_length, 1))
    for i in range(side_length):
        S[i] = 0.04
    phi = 0.2
    discount = 0.99
    return {
        "orchard_length": side_length,
        "num_agents": num_agents,
        "S": S,
        "phi": phi,
        "discount": discount
    }


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "policyitchk"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
