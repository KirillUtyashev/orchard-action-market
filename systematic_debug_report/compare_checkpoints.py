"""Compare two model checkpoint .pt files parameter by parameter.

Usage:
    python compare_checkpoints.py path/to/step_1.pt path/to/step_1.pt

Handles both checkpoint formats:
  old: {"networks": [state_dict, ...]}
  new: {"algorithm": "value", "step": N, "critics": [state_dict, ...]}

Prints max abs diff per parameter and flags any that exceed the tolerance.
"""

from __future__ import annotations

import sys
import torch


TOL = 1e-7


def load_state_dicts(path: str) -> list[dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "networks" in ckpt:
        return ckpt["networks"]
    if "critics" in ckpt:
        return ckpt["critics"]
    raise ValueError(f"Unknown checkpoint format in {path}: keys={list(ckpt.keys())}")


def compare(path_a: str, path_b: str) -> bool:
    """Return True if all params match within TOL."""
    nets_a = load_state_dicts(path_a)
    nets_b = load_state_dicts(path_b)

    if len(nets_a) != len(nets_b):
        print(f"MISMATCH: {path_a} has {len(nets_a)} networks, {path_b} has {len(nets_b)}")
        return False

    all_match = True
    for net_idx, (sd_a, sd_b) in enumerate(zip(nets_a, nets_b)):
        keys_a = set(sd_a.keys())
        keys_b = set(sd_b.keys())
        if keys_a != keys_b:
            print(f"  Network {net_idx}: key mismatch")
            print(f"    only in A: {keys_a - keys_b}")
            print(f"    only in B: {keys_b - keys_a}")
            all_match = False
            continue

        first_fail: str | None = None
        max_diff_name = ""
        max_diff_val = 0.0
        for name in sorted(sd_a.keys()):
            t_a = sd_a[name].float()
            t_b = sd_b[name].float()
            if t_a.shape != t_b.shape:
                print(f"  Network {net_idx} param {name}: shape {t_a.shape} vs {t_b.shape}")
                all_match = False
                continue
            diff = (t_a - t_b).abs().max().item()
            if diff > max_diff_val:
                max_diff_val = diff
                max_diff_name = name
            if diff > TOL and first_fail is None:
                first_fail = name

        status = "OK" if first_fail is None else f"FAIL (first mismatch: {first_fail})"
        print(f"  Network {net_idx}: max_diff={max_diff_val:.2e} at '{max_diff_name}' — {status}")
        if first_fail is not None:
            all_match = False

    return all_match


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <checkpoint_a.pt> <checkpoint_b.pt>")
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]
    print(f"Comparing:\n  A: {path_a}\n  B: {path_b}\n")
    match = compare(path_a, path_b)
    print()
    if match:
        print("PASS: all parameters match within tolerance 1e-7")
        sys.exit(0)
    else:
        print("FAIL: parameters differ")
        sys.exit(1)


if __name__ == "__main__":
    main()
