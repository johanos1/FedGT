from matplotlib import pyplot as plt
import json
import numpy as np
from itertools import product


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))

data = ["mnist"]  # , "cifar10"]
MC_iter = 10
alpha_list = [np.inf]
epochs_list = [1]
n_malicious_list = [5]
batch_size_list = [16, 32, 64, 128, 256]

foldername = "20221219_163545/"

sim_params = list(
    product(data, alpha_list, epochs_list, n_malicious_list, batch_size_list)
)

for k, (d, a, e, m, bs) in enumerate(sim_params):

    # load file
    prefix = f"./results/" + foldername + f"{d.upper()}_"
    suffix = f"m-{m}_e-{e}_bs-{bs}_alpha-{a}_totalMC-{MC_iter}"
    suffix_0 = suffix + f"_MODE-0.txt"
    suffix_1 = suffix + f"_MODE-1.txt"
    suffix_2 = suffix + f"_MODE-2.txt"

    try:
        data_no_defence = json.load(open(prefix + suffix_0))  # no defence
        acc_no_defence = np.array(data_no_defence["accuracy"]).squeeze()
    except:
        acc_no_defence = []

    try:
        data_oracle = json.load(open(prefix + suffix_1))  # oracle
        acc_oracle = np.array(data_oracle["accuracy"]).squeeze()
    except:
        acc_oracle = []

    data = json.load(open(prefix + suffix_2))  # group testing
    acc = np.array(data["accuracy"])
    acc = acc[0]
    # extract simulation data
    group_acc = np.array(data["group_acc"])
    syndrome = np.array(data["syndrome"])
    DEC = np.array(data["DEC"]).astype(bool)
    P_MD = np.array(data["P_MD"])
    P_FA = np.array(data["P_FA"])
    threshold_vec = np.array(data["threshold_vec"])
    n_clients = data["client_number"]
    n_malicious = data["n_malicious"]
    comm_rounds = data["comm_round"]
    malicious_clients = np.array(data["malicious_clients"]).astype(bool)

    plot_prefix = f"{d.upper()} with {MC_iter} MC iterations"
    fig.suptitle(plot_prefix)
    # FIG 1:
    # average accuracy over communication rounds
    # acc is structured as: thresholds x MC iter x comm rounds
    average_acc = np.mean(acc, axis=1)
    str_legend = []
    for i, threshold in enumerate(threshold_vec):
        ax[0].plot(range(comm_rounds), average_acc[i, :])
        str_legend.append(str("{0:.2f}".format(threshold)))
    ax[0].plot(range(comm_rounds), np.mean(acc_oracle, axis=0), "--b")
    str_legend.append("oracle")
    ax[0].plot(range(comm_rounds), np.mean(acc_no_defence, axis=0), "--r")
    str_legend.append("no defence")
    ax[0].set_title("Average accuracy")
    ax[0].set_xlabel("Commuication rounds")
    ax[0].set_ylabel("Accuracy")
    ax[0].xaxis.grid(True)
    ax[0].yaxis.grid(True)
    ax[0].legend(str_legend)

    plt.show
