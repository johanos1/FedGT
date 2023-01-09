from matplotlib import pyplot as plt

"""
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)
"""
import json
import numpy as np
from itertools import product


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 5))

data = ["mnist"]  # , "cifar10"]
MC_iter = 100
alpha_list = [np.inf]
epochs_list = [1]
n_malicious_list = [4]
batch_size_list = [64]

foldername = "20221221_213909/"
foldername0 = "20221221_213332/"
foldername1 = "20221221_213421/"

sim_params = list(
    product(data, alpha_list, epochs_list, n_malicious_list, batch_size_list)
)

for k, (d, a, e, m, bs) in enumerate(sim_params):

    # load file
    prefix = f"./results/" + foldername + f"{d.upper()}_"
    prefix0 = f"./results/" + foldername0 + f"{d.upper()}_"
    prefix1 = f"./results/" + foldername1 + f"{d.upper()}_"
    suffix = f"m-{m}_e-{e}_bs-{bs}_alpha-{a}_totalMC-{MC_iter}"
    suffix_0 = suffix + f"_MODE-0.txt"
    suffix_1 = suffix + f"_MODE-1.txt"
    # suffix_0 = f"m-{m}_e-{e}_bs-{bs}_alpha-{a}_totalMC-100" + f"_MODE-0.txt"
    # suffix_1 = f"m-{m}_e-{e}_bs-{bs}_alpha-{a}_totalMC-100" + f"_MODE-1.txt"
    suffix_2 = suffix + f"_MODE-2.txt"

    try:
        data_no_defence = json.load(open(prefix0 + suffix_0))  # no defence
        acc_no_defence = np.array(data_no_defence["accuracy"]).squeeze()
    except:
        acc_no_defence = []

    try:
        data_oracle = json.load(open(prefix1 + suffix_1))  # oracle
        acc_oracle = np.array(data_oracle["accuracy"]).squeeze()
    except:
        acc_oracle = []

    data = json.load(open(prefix + suffix_2))  # group testing
    acc = np.array(data["accuracy"])
    # extract simulation data
    group_acc = np.array(data["group_acc"])
    syndrome = np.array(data["syndrome"])
    DEC = np.array(data["DEC"]).astype(bool)
    syndrome = np.array(data["syndrome"])
    P_MD = np.array(data["P_MD"])
    P_FA = np.array(data["P_FA"])
    threshold_vec = np.array(data["threshold_vec"])
    n_clients = data["client_number"]
    n_malicious = data["n_malicious"]
    comm_rounds = data["comm_round"]
    try:
        cross_prop = data["cross_prop"]
        # cross_prop = cross_prop[0]
    except:
        cross_prop = []

    if not bool(cross_prop):
        cross_prop = 0.05
    malicious_clients = np.array(data["malicious_clients"]).astype(bool)

    plot_prefix = f"{d.upper()} with {MC_iter} MC iterations for {m} malicious nodes - Chm crossover prop {cross_prop}"
    fig.suptitle(plot_prefix)
    # FIG 1:
    # average accuracy over communication rounds
    # acc is structured as: thresholds x MC iter x comm rounds
    average_acc = np.mean(acc, axis=1)
    str_legend = []
    for i, threshold in enumerate(threshold_vec):
        ax[0, 0].plot(range(comm_rounds), average_acc[i, :])
        str_legend.append(str("{0:.2f}".format(threshold)))
    ax[0, 0].plot(range(comm_rounds), np.mean(acc_oracle, axis=0), "--b")
    str_legend.append("oracle")
    ax[0, 0].plot(range(comm_rounds), np.mean(acc_no_defence, axis=0), "--r")
    str_legend.append("no defence")
    ax[0, 0].set_title("Average accuracy")
    ax[0, 0].set_xlabel("Communication rounds")
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
    # find out how many malign/benign users are included
    benign_users_included = []
    malign_users_included = []
    for i, _ in enumerate(threshold_vec):
        benign_users = 0
        malign_users = 0
        for j in range(MC_iter):
            benign_indx = np.where(malicious_clients[i, j, :] == False)[0]
            malicious_indx = np.where(malicious_clients[i, j, :] == True)[0]
            benign_users += np.sum(DEC[i, j, benign_indx] == False)
            malign_users += np.sum(DEC[i, j, malicious_indx] == True)
        avg_benign_included = (benign_users / MC_iter) / (n_clients - n_malicious)
        avg_malign_included = (malign_users / MC_iter) / (n_malicious)
        benign_users_included.append(avg_benign_included)
        malign_users_included.append(avg_malign_included)

    ax[1, 0].plot(threshold_vec, final_acc)
    ax[1, 0].plot(
        threshold_vec, np.ones(threshold_vec.shape) * np.mean(acc_oracle[:, -1]), "--b"
    )
    ax[1, 0].plot(
        threshold_vec,
        np.ones(threshold_vec.shape) * np.mean(acc_no_defence[:, -1]),
        "--r",
    )
    # ax[1, 0].plot(threshold_vec, benign_users_included)
    # ax[1, 0].plot(threshold_vec, malign_users_included)
    ax[1, 0].set_title(f"Average accuracy @ {comm_rounds} rounds")
    ax[1, 0].set_xlabel("BCJR Threshold")
    ax[1, 0].xaxis.grid(True)
    ax[1, 0].yaxis.grid(True)
    ax[1, 0].legend(["group test", "oracle", "no defence"])

    # FIG 4:
    # median accuracy
    median_acc = np.median(acc, axis=1)[:, -1]
    ax[1, 1].plot(threshold_vec, median_acc)
    ax[1, 1].plot(
        threshold_vec, np.ones(threshold_vec.shape) * np.mean(acc_oracle[:, -1]), "--b"
    )
    ax[1, 1].plot(
        threshold_vec,
        np.ones(threshold_vec.shape) * np.mean(acc_no_defence[:, -1]),
        "--r",
    )
    ax[1, 1].set_title(f"Median accuracy @ {comm_rounds} rounds")
    ax[1, 1].set_xlabel("BCJR Threshold")
    ax[1, 1].set_ylabel("Accuracy")
    ax[1, 1].xaxis.grid(True)
    ax[1, 1].yaxis.grid(True)
    ax[1, 0].legend(["group test", "oracle", "no defence"])
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
    )
    plt.savefig("foo.pdf", bbox_inches="tight")
    plt.show()
