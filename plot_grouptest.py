from matplotlib import pyplot as plt
import json
import numpy as np
from itertools import product


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 5))

data = ["mnist", "cifar"]
alpha_list = [np.inf]
epochs_list = [1]
n_malicious_list = [5]
batch_size_list = [64]

sim_params = list(
    product(data, alpha_list, epochs_list, n_malicious_list, batch_size_list)
)

for k, (d, a, e, m, bs) in enumerate(sim_params):

    # load file
    if d == "mnist":
        prefix = "./results/MNIST_"
        plot_prefix = "MNIST"
    elif d == "cifar":
        prefix = "./results/CIFAR10_"
        plot_prefix = "CIFAR10"
    suffix = f"m-{m}_e-{e}_bs-{bs}_alpha-{a}.txt"
    sim_title = prefix + suffix

    f = open(sim_title)
    data = json.load(f)

    # extract simulation data
    group_acc = np.array(data["group_acc"])
    acc = np.array(data["accuracy"])
    syndrome = np.array(data["syndrome"])
    DEC = np.array(data["DEC"])
    syndrome = np.array(data["syndrome"])
    P_MD = np.array(data["P_MD"])
    P_FA = np.array(data["P_FA"])
    threshold_vec = np.array(data["threshold_vec"])
    MC_iter = data["total_MC_it"]
    n_clients = data["client_number"]
    comm_rounds = data["comm_round"]

    fig.suptitle(plot_prefix)
    # FIG 1:
    # average accuracy over communication rounds
    # acc is structured as: thresholds x MC iter x comm rounds
    average_acc = np.mean(acc, axis=1)
    str_legend = []
    for i, threshold in enumerate(threshold_vec):
        ax[0, 0].plot(range(comm_rounds), average_acc[i, :])
        str_legend.append(str(threshold))
    ax[0, 0].set_title("Average accuracy")
    ax[0, 0].set_xlabel("Commuication rounds")
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 0].xaxis.grid(True)
    ax[0, 0].yaxis.grid(True)
    ax[0, 0].legend(str_legend)

    # FIG 2:
    # P_FA vs 1-P_MD
    ax[0, 1].plot(P_FA, 1 - P_MD)
    ax[0, 1].set_title("ROC")
    ax[0, 1].set_xlabel("P_FA")
    ax[0, 1].set_ylabel("1-P_MD")
    ax[0, 1].xaxis.grid(True)
    ax[0, 1].yaxis.grid(True)

    # FIG 3:
    # average accuracy wrt threshold after comm rounds
    final_acc = average_acc[:, -1]
    ax[1, 0].set_title(f"Average accuracy @ {comm_rounds} rounds")
    ax[1, 0].set_xlabel("GL Threshold")
    ax[1, 0].set_ylabel("Accuracy")
    ax[1, 0].xaxis.grid(True)
    ax[1, 0].yaxis.grid(True)

    # FIG 4:
    # median accuracy
    median_acc = np.median(acc, axis=1)[-1]
    ax[1, 1].set_title(f"Median accuracy @ {comm_rounds} rounds")
    ax[1, 1].set_xlabel("GL Threshold")
    ax[1, 1].set_ylabel("Accuracy")
    ax[1, 1].xaxis.grid(True)
    ax[1, 1].yaxis.grid(True)

    plt.show
