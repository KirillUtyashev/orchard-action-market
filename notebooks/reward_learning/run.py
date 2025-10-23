import matplotlib.pyplot as plt
import sys
import os

sys.path.append("../")

from config import FINAL_DIR, OUT_DATA_DIR
import json

# create 3 plots: 6x6, 9x9, 12x12 based on data foldr
# each plot has 4 lines: cnn centralized, cnn decentralized, mlp centralized, mlp decentralized
# x-axis is hidden features: [8, 32, 64, 256]
# y-axis is accuracy


GRID_SIZES = ["6x6", "9x9", "12x12"]

# X-axis values for the plots.
HIDDEN_FEATURES = [8, 32, 64, 256]

# This dictionary maps the grid size to the specific agent information in the filenames.
GRID_TO_AGENT_MAP = {"6x6": "2_agents", "9x9": "4_agents", "12x12": "7_agents"}

# Define the different data series you want to plot.
# The dictionary key will be used as the label in the plot legend.
# The value is the prefix of the corresponding data file.
# Note: Based on your image, there are no files for "cnn decentralized".
# This script plots all four available series from your 'data' folder.
SERIES_TO_PLOT = {
    "CNN Centralized": "cnn_centralized_accuracies",
    "MLP Centralized": "mlp_centralized_accuracies",
    "CNN Decentralized (1 Acting)": "cnn_decentralized_accuracies_1acting",
    "MLP Decentralized (1 Acting)": "mlp_decentralized_accuracies_1acting",
    "CNN Decentralized (All Acting)": "cnn_decentralized_accuracies_allActing",
    "MLP Decentralized (All Acting)": "mlp_decentralized_accuracies_allActing",
}

# --- Plotting Function ---


def load_and_plot():
    """
    Loads accuracy data from text files and generates a plot for each grid size.
    """
    # Create a separate plot for each grid size.
    for grid in GRID_SIZES:
        fig, ax = plt.subplots(figsize=(12, 7))

        print(f"--- Generating plot for {grid} grid size ---")

        # Iterate through the models/series defined in the configuration.
        for label, file_prefix in SERIES_TO_PLOT.items():
            # Construct the full filename dynamically.
            agent_info = GRID_TO_AGENT_MAP[grid]
            filename = f"{file_prefix}_{grid}_{agent_info}.txt"
            filepath = os.path.join(OUT_DATA_DIR, filename)

            # Load data with error handling.
            try:
                with open(filepath, "r") as f:
                    accuracies = json.load(f)

                # Ensure the data has the expected length before plotting.
                if len(accuracies) == len(HIDDEN_FEATURES):
                    ax.plot(
                        HIDDEN_FEATURES,
                        accuracies,
                        marker="o",
                        linestyle="-",
                        label=label,
                    )
                    print(f"  ✓ Plotted '{label}' from {filename}")
                else:
                    print(f"  ! Warning: Data length mismatch in {filename}. Skipping.")

            except FileNotFoundError:
                print(f"  ! Warning: File not found: {filename}. Skipping this line.")
            except json.JSONDecodeError:
                print(
                    f"  ! Warning: Could not parse JSON from {filename}. Skipping this line."
                )

        # --- Formatting the plot ---
        ax.set_title(f"Model Accuracy vs. Hidden Features ({grid} Grid)", fontsize=16)
        ax.set_xlabel("Number of Hidden Features", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)

        # Use a log scale for the x-axis to better space out the feature counts.
        ax.set_xticks(HIDDEN_FEATURES)

        ax.grid(True, which="both", ls="--", c="0.7")
        ax.legend()

        save_filename = f"model_accuracy_{grid}.png"
        save_path = os.path.join(FINAL_DIR, save_filename)

        # Use bbox_inches='tight' to prevent labels from being cut off
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  ✓ Plot saved to {save_path}")
        plt.close(fig)


# --- Run the script ---
if __name__ == "__main__":
    load_and_plot()
