from pathlib import Path
import numpy as np
import torch


def get_config():
    discount = 0.99
    return {
        "discount": discount,
    }


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "policyitchk"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
