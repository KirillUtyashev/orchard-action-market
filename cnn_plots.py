from plots import add_to_plots, plot_hybrid_smoothed

from pathlib import Path

import matplotlib.pyplot as plt


def plot_reward_learning_progress(name: str, network):
    """
    Generates and saves plots specific to a reward learning experiment.

    Args:
        name: The name of the experiment for saving files.
        network: The trained network object (e.g., RewardCNN), which contains
                 the loss_history and model parameters.
    """
    # Create the output directory
    graph_folder = Path("graphs")
    name_folder = graph_folder / name
    name_folder.mkdir(parents=True, exist_ok=True)

    # --- 1. Plot the Training Loss ---
    # We get the loss history directly from the network object.
    loss_history = network.loss_history
    if loss_history:
        fig = plot_hybrid_smoothed(
            [loss_history],
            labels=["Training MSE Loss"],
            title="Reward Prediction Training Loss",
            xlabel="Training Step",
            ylabel="MSE Loss",
        )
        # Use log scale for loss plots
        ax = fig.gca()
        ax.set_yscale("log")
        ax.grid(True)
        fig.savefig(name_folder / f"Training_Loss.png")
        plt.close(fig)
