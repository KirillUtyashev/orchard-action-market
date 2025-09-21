from pathlib import Path
import numpy as np
import torch


def get_config():  # FIXME this doesn't belong here. It should be in
    # command line args or we should entirely switch to hydra yaml files.
    discount = 0.99
    return {
        "discount": discount,
    }


PROJECT_ROOT = Path(__file__).resolve().parent


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- PyTorch is configured to use: {DEVICE} ---")

OUT_DIR = PROJECT_ROOT / "out"
OUT_DIR.mkdir(exist_ok=True)
LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
GRAPHS_DIR = OUT_DIR / "graphs"
GRAPHS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
