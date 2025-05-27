import matplotlib.pyplot as plt


def setup_plots(dictn, plot):
    for param_tensor in dictn:
        print(dictn[param_tensor].size())
        plot[param_tensor] = []
        if dictn[param_tensor].dim() > 1 and "weight" in param_tensor:  # and "1" not in param_tensor:
            for id in range(5):
                plot[param_tensor + str(id)] = []


def add_to_plots(dictn, timestep, plot):
    for param_tensor in dictn:
        if dictn[param_tensor].dim() > 1 and "weight" in param_tensor:  # and "1" not in param_tensor:
            for id in range(5):
                plot[param_tensor + str(id)].append(dictn[param_tensor].cpu().flatten()[id])


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


def graph_plots(dictn, name, plot, loss_plot, loss_plot1, loss_plot2, ratio_plot):
    global graph
    graph += 1

    # --- parameter trajectories (unchanged) ---
    plt.figure("plots" + str(graph) + name, figsize=(10, 5))
    num = 0
    for param_tensor in dictn:
        num += 1
        if num == 10:
            break
        if dictn[param_tensor].dim() > 1 and "weight" in param_tensor:
            for id in range(5):
                if id == 0:
                    plt.plot(plot[param_tensor + str(id)], color=colours[num], label="Tensor " + str(num))
                else:
                    plt.plot(plot[param_tensor + str(id)], color=colours[num])
    plt.legend()
    plt.title("Model Parameters during Training, iteration " + str(graph))
    plt.savefig("Params_" + name + str(graph) + ".png")
    plt.close()

    # --- loss curves ---
    plt.figure("loss" + str(graph), figsize=(10, 5))
    plt.plot(loss_plot)
    plt.plot(loss_plot1)
    plt.plot(loss_plot2)
    plt.title("Value Function for Sample State, iteration " + str(graph))
    plt.savefig("Val_" + name + str(graph) + ".png")
    plt.close()

    # --- ratio plot (new) ---
    plt.figure("ratio" + str(graph), figsize=(10, 5))
    plt.plot(ratio_plot)
    plt.title("Ratio over Time, iteration " + str(graph))
    plt.xlabel("Training Step")
    plt.ylabel("Ratio")
    plt.savefig("Ratio_" + name + str(graph) + ".png")
    plt.close()




