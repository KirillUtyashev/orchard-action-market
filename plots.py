from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def setup_plots(dictn, plot):
    for param_tensor in dictn:
        print(dictn[param_tensor].size())
        plot[param_tensor] = []
        if dictn[param_tensor].dim() > 1 and "weight" in param_tensor:  # and "1" not in param_tensor:
            for id in range(5):
                plot[param_tensor + str(id)] = []


def add_to_plots(dictn, plot):
    for param_tensor in dictn:
        if dictn[param_tensor].dim() > 1 and "weight" in param_tensor:
            tensor = dictn[param_tensor].cpu().flatten()
            # Take weights from different parts of the tensor
            indices = np.linspace(0, len(tensor)-1, 5, dtype=int)
            for idx, sample_idx in enumerate(indices):
                if param_tensor + str(idx) not in plot.keys():
                    plot[param_tensor + str(idx)] = []
                plot[param_tensor + str(idx)].append(tensor[sample_idx].item())


def init_plots():
    one_plot = {}
    two_plot = {}
    loss_plot = []
    loss_plot1 = []
    loss_plot2 = []
    ratio_plot = []
    return one_plot, two_plot, loss_plot, loss_plot1, loss_plot2, ratio_plot


colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
graph = 0


def graph_plots(name, plot, critic_loss, loss_plot, loss_plot1, loss_plot2, v_weights_plot=None):
    global graph
    graph += 1
    graph_folder = Path("graphs")
    graph_folder.mkdir(parents=True, exist_ok=True)

    name_folder = graph_folder / name
    name_folder.mkdir(parents=True, exist_ok=True)

    # --- parameter trajectories ---
    plt.figure(f"plots_{graph}_{name}", figsize=(10, 5))

    # Get unique layer names (without the index suffix)
    layers = set()
    for key in plot.keys():
        # Extract base name without the numeric suffix
        base_name = ''.join(c for c in key)
        layers.add(base_name[:len(base_name) - 1])

    # Plot each layer's sampled weights with consistent colors
    for idx, layer_name in enumerate(sorted(layers)):
        if idx >= 10:  # Limit to 10 layers for readability
            break

        # Get all samples for this layer
        samples = [plot[f"{layer_name}{i}"] for i in range(5)]

        # Plot each sample trajectory
        for sample_idx, sample_data in enumerate(samples):
            if sample_idx == 0:
                plt.plot(sample_data, color=colours[idx], label=f"Layer {idx+1}")
            else:
                plt.plot(sample_data, color=colours[idx], alpha=0.5)

    plt.legend()
    plt.title(f"Model Parameters during Training, iteration {graph}")
    plt.xlabel("Training Step")
    plt.ylabel("Parameter Value")
    plt.savefig(name_folder / f"Params_{name}_{graph}.png")
    plt.close()

    # --- Sample state evaluation curves ---
    plt.figure(f"loss_{graph}", figsize=(10, 5))
    plt.plot(loss_plot, label="State 1")
    plt.plot(loss_plot1, label="State 2")
    plt.plot(loss_plot2, label="State 3")
    plt.legend()
    plt.title(f"Value Function for Sample States, iteration {graph}")
    plt.xlabel("Training Step")
    plt.ylabel("Value")
    plt.savefig(name_folder / f"Val_{name}_{graph}.png")
    plt.close()

    # --- Loss curves ---
    plt.figure(f"training_loss_{graph}", figsize=(10, 5))
    plt.plot(critic_loss, label="Training MSE Loss")
    plt.title(f"Training Loss over Time, iteration {graph}")
    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    plt.yscale('log')  # Often helpful for loss plots
    plt.grid(True)
    plt.savefig(name_folder / f"Training_Loss_{name}_{graph}.png")
    plt.close()

    if len(v_weights_plot) > 0:
        plt.figure(f"plots_{graph}_{name}", figsize=(10, 5))

        # Get unique layer names (without the index suffix)
        layers = set()
        for key in plot.keys():
            # Extract base name without the numeric suffix
            base_name = ''.join(c for c in key)
            layers.add(base_name[:len(base_name) - 1])

        # Plot each layer's sampled weights with consistent colors
        for idx, layer_name in enumerate(sorted(layers)):
            if idx >= 10:  # Limit to 10 layers for readability
                break

            # Get all samples for this layer
            samples = [plot[f"{layer_name}{i}"] for i in range(5)]

            # Plot each sample trajectory
            for sample_idx, sample_data in enumerate(samples):
                if sample_idx == 0:
                    plt.plot(sample_data, color=colours[idx], label=f"Layer {idx+1}")
                else:
                    plt.plot(sample_data, color=colours[idx], alpha=0.5)

        plt.legend()
        plt.title(f"Model Parameters during Training, Critic, iteration {graph}")
        plt.xlabel("Training Step")
        plt.ylabel("Parameter Value")
        plt.savefig(name_folder / f"Value_{name}_{graph}.png")
        plt.close()

