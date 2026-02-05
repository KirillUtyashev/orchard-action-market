# mae_vs_nn_size_plotter.py

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _extract_first_int(s: str) -> Optional[int]:
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def _load_errors_json(path: Path) -> Dict[str, List[float]]:
    with open(path, "r") as f:
        payload = json.load(f)
    # Supports payload = {"errors_by_state": {...}} or payload = {...}
    return payload["errors_by_state"] if "errors_by_state" in payload else payload


def _mae_from_ape(ape_by_state: Dict[str, List[float]]) -> float:
    all_apes: List[float] = []
    for vals in ape_by_state.values():
        all_apes.extend(vals)

    if not all_apes:
        return float("nan")

    arr = np.asarray(all_apes, dtype=np.float64)

    # APE values are already in percent units (e.g., 12.3 == 12.3%)
    return float(np.mean(np.abs(arr)))  # [code_file:0]


def plot_mae_vs_nn_size_for_picker_r(
        data_dir: Path,
        picker_r: str,
        *,
        json_filename: str = "final_eval_errors.json",
        input_dims: Tuple[str, str] = ("3", "326"),
        output_png: Optional[Path] = None,
        output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Plot MAE vs NN size for two input dimensions under supervised/{picker_r}.

    Expected folder structure:
        {data_dir}/supervised/{picker_r}/{input_dim}/{nn_size_folder}/.../{json_filename}

    Notes:
    - NN size is inferred from the first integer in the folder name containing the json file.
    - If multiple files exist per (input_dim, nn_size) (e.g., multiple seeds), MAE is averaged.

    Returns:
        DataFrame with columns: picker_r, input_dim, nn_size, mae
    """
    base = Path(data_dir) / "supervised" / str(picker_r)
    rows = []

    for dim in input_dims:
        dim_dir = base / str(dim)
        if not dim_dir.exists():
            continue

        for json_path in dim_dir.rglob(json_filename):
            nn_folder = json_path.parent
            nn_size = _extract_first_int(nn_folder.name)
            if nn_size is None:
                nn_size = _extract_first_int(nn_folder.parent.name)

            errors_by_state = _load_errors_json(json_path)
            mae = _mae_from_ape(errors_by_state)

            rows.append(
                {
                    "picker_r": str(picker_r),
                    "input_dim": int(dim),
                    "nn_size": nn_size,
                    "mae": mae,
                    "json_path": str(json_path),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"No '{json_filename}' files found under {base} for input_dims={input_dims}."
        )

    df = df.dropna(subset=["nn_size"]).copy()
    df["nn_size"] = df["nn_size"].astype(int)

    df_agg = (
        df.groupby(["picker_r", "input_dim", "nn_size"], as_index=False)["mae"]
        .mean()
        .sort_values(["input_dim", "nn_size"])
    )

    # Plot two curves (one per input dimension)
    plt.figure(figsize=(7, 4.5))
    for dim in sorted(df_agg["input_dim"].unique()):
        sub = df_agg[df_agg["input_dim"] == dim].sort_values("nn_size")
        plt.plot(sub["nn_size"], sub["mae"], marker="o", linewidth=2, label=f"input_dim={dim}")

    plt.xlabel("NN size")
    plt.ylabel("MAE")
    plt.title(f"MAE vs NN size (picker_r={picker_r})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_png is None:
        output_png = base / f"mae_vs_nn_size_picker_r_{picker_r}.png"
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    if output_csv is None:
        output_csv = base / f"mae_vs_nn_size_picker_r_{picker_r}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(output_csv, index=False)

    return df_agg


def plot_variance(data_dir, variances):
    base = Path(data_dir) / "supervised" / "-1" / "3"
    rows = []

    for var in variances:
        path = base / str(var) / "final_eval_errors_16_1000.json"
        errors_by_state = _load_errors_json(path)
        mae = _mae_from_ape(errors_by_state)
        rows.append({"var": var, "mae": mae})

    df = pd.DataFrame(rows).sort_values("var")

    plt.figure(figsize=(7, 4.5))

    # One curve, points connected by straight dotted segments
    plt.plot(
        df["var"], df["mae"],
        marker="o",
        linestyle=":",  # ":" is matplotlib's dotted linestyle [code_file:0]
        linewidth=2,
        label="MAE"
    )

    plt.xlabel("Variance")
    plt.ylabel("MAE % of True Value")
    plt.title("MAE vs Variance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    from config import data_dir
    # plot_variance(data_dir, [0.0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2])
    plot_variance(data_dir, [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
