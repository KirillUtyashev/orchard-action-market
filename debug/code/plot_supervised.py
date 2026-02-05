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


def _pick_error_path(base: Path, var: int | float, alpha: float, scheduled: bool) -> Path:
    """Return the first existing path among known naming patterns."""
    var_dir = base / str(var)

    candidates = []
    if scheduled:
        candidates += [
            var_dir / f"final_eval_errors_16_1000_{alpha}_True.json",
            var_dir / f"final_eval_errors_16_1000_{alpha}_true.json",
            var_dir / f"final_eval_errors_16_1000_{alpha}_{True}.json",
<<<<<<< HEAD
            ]
=======
        ]
>>>>>>> 7d9a105 (Plots)
    else:
        candidates += [
            var_dir / f"final_eval_errors_16_1000_{alpha}.json",
            var_dir / f"final_eval_errors_16_1000_{alpha}_False.json",
            var_dir / f"final_eval_errors_16_1000_{alpha}_false.json",
            var_dir / f"final_eval_errors_16_1000_{alpha}_{False}.json",
<<<<<<< HEAD
            ]
=======
        ]
>>>>>>> 7d9a105 (Plots)

    for p in candidates:
        if p.exists():  # Path.exists() checks if the filesystem path exists. [web:3]
            return p

    raise FileNotFoundError(
        f"No eval error file found for var={var}, alpha={alpha}, scheduled={scheduled}. "
        f"Tried: {[str(p) for p in candidates]}"
    )


def plot_variance(data_dir, variances, alphas):
    base = Path(data_dir) / "supervised" / "-1" / "3"
    variances = sorted(set(variances))

    fig, ax = plt.subplots(figsize=(8, 4.8))

    def plot_series(alpha: float, scheduled: bool, *, linestyle: str, marker: str, label_suffix: str = ""):
        rows = []
        for var in variances:
            path = _pick_error_path(base, var, alpha, scheduled)
            errors_by_state = _load_errors_json(path)
            mae = _mae_from_ape(errors_by_state)
            rows.append({"var": var, "mae": mae})

        df = pd.DataFrame(rows).sort_values("var")
        ax.plot(
            df["var"], df["mae"],
            marker=marker,
            markersize=5.5,
            linestyle=linestyle,
            linewidth=2.2,
            label=f"lr={alpha:g}{label_suffix}",
        )

    # Unscheduled runs
    for alpha in alphas:
        plot_series(alpha, scheduled=False, linestyle=":", marker="o")

    # Scheduled runs (keep your existing choice; feel free to parameterize)
    for alpha in [0.001, 0.0001]:
        plot_series(alpha, scheduled=True, linestyle="-", marker="s", label_suffix=" (sched)")

    # ---- unclump x-axis (supports 0) ----
    pos = [v for v in variances if v > 0]
    linthresh = min(pos) if pos else 1.0
    ax.set_xscale("symlog", linthresh=linthresh)  # symlog has a linear region set by linthresh. [web:65]

    ax.set_xlim(min(variances), max(variances))
    if len(variances) <= 12:
        ax.set_xticks(variances)
    else:
        ax.set_xticks([0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="x")
    # -----------------------------------

    ax.set_xlabel("Variance")
    ax.set_ylabel("MAE % of True Value")
    ax.set_title("MAE vs Variance (by Learning Rate)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True, framealpha=0.9, ncols=2)

    fig.tight_layout()
    plt.savefig(base / "main.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

<<<<<<< HEAD

def _pick_error_path_nn(base: Path, var: int | float, nn_size: int, *, alpha: float = 0.001) -> Path:
    """Return the first existing path among known naming patterns for NN-size runs."""
    var_dir = base / str(var)

    # You said: replace "16" with NN size, and the later part should be "0.001_True"
    candidates = [
        var_dir / f"final_eval_errors_{nn_size}_1000_{alpha}_True.json",
        var_dir / f"final_eval_errors_{nn_size}_1000_{alpha}_true.json",
        var_dir / f"final_eval_errors_{nn_size}_1000_{alpha}_{True}.json",
        ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"No eval error file found for var={var}, nn_size={nn_size} (alpha={alpha}, scheduled=True). "
        f"Tried: {[str(p) for p in candidates]}"
    )


def plot_variance_by_nn_size(data_dir, variances, nn_sizes, *, alpha: float = 0.001, out_name: str = "main_by_nn.png"):
    base = Path(data_dir) / "supervised" / "-1" / "3"
    variances = sorted(set(variances))
    nn_sizes = sorted(set(nn_sizes))

    fig, ax = plt.subplots(figsize=(8, 4.8))

    # simple style cycling (optional)
    markers = ["o", "s", "D", "^", "v", "P", "X"]
    linestyles = ["-", "--", "-.", ":"]

    def plot_series(nn_size: int, *, linestyle: str, marker: str):
        rows = []
        for var in variances:
            path = _pick_error_path_nn(base, var, nn_size, alpha=alpha)
            errors_by_state = _load_errors_json(path)
            mae = _mae_from_ape(errors_by_state)
            rows.append({"var": var, "mae": mae})

        df = pd.DataFrame(rows).sort_values("var")
        ax.plot(
            df["var"], df["mae"],
            marker=marker,
            markersize=5.5,
            linestyle=linestyle,
            linewidth=2.2,
            label=f"hid_dim={nn_size}",
        )

    for i, nn_size in enumerate(nn_sizes):
        plot_series(
            nn_size,
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
        )

    # ---- unclump x-axis (supports 0) ----
    pos = [v for v in variances if v > 0]
    linthresh = min(pos) if pos else 1.0
    ax.set_xscale("symlog", linthresh=linthresh)  # linear region is (-linthresh, linthresh) [web:2]

    ax.set_xlim(min(variances), max(variances))
    if len(variances) <= 12:
        ax.set_xticks(variances)
    else:
        ax.set_xticks([0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.ticklabel_format(style="plain", axis="x")
    # -----------------------------------

    ax.set_xlabel("Variance")
    ax.set_ylabel("MAE % of True Value")
    ax.set_title("MAE vs Variance (by NN size)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True, framealpha=0.9, ncols=2)

    fig.tight_layout()
    plt.savefig(base / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

=======
>>>>>>> 7d9a105 (Plots)

if __name__ == "__main__":
    from config import data_dir
    # plot_variance(data_dir, [0.0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2])
<<<<<<< HEAD
    # plot_variance(data_dir, [0.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0], [0.01, 0.001, 0.0001])
    plot_variance_by_nn_size(data_dir, [0.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0], [4, 8, 16, 32, 64, 128])

=======
    plot_variance(data_dir, [0.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0], [0.01, 0.001, 0.0001])
>>>>>>> 7d9a105 (Plots)
