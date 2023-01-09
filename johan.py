from matplotlib import pyplot as plt
import json
import numpy as np
from itertools import product


fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 5))

data = ["mnist"]  # , "cifar10"]
if data[0] == "cifar10":
    MC_iter = 10
    epochs_list = [5]
    n_malicious_list = [4]
    batch_size_list = [128]
elif data[0] == "mnist":
    MC_iter = 100
    epochs_list = [1]
    n_malicious_list = [4]
    batch_size_list = [128]

alpha_list = [np.inf]
sim_params = list(
    product(data, alpha_list, epochs_list, n_malicious_list, batch_size_list)
)

foldername = "20221221_213909/"
foldername0 = "20221221_213332/"
foldername1 = "20221221_213421/"

for k, (d, a, e, m, bs) in enumerate(sim_params):

    # prefix = f"./results/" + foldername + f"{d.upper()}_"
    # prefix0 = f"./results/" + foldername0 + f"{d.upper()}_"
    # prefix1 = f"./results/" + foldername1 + f"{d.upper()}_"
    prefix = f"./results/{d.upper()}_"
    prefix0 = f"./results/{d.upper()}_"
    prefix1 = f"./results/{d.upper()}_"
    suffix = f"m-{m}_e-{e}_bs-{bs}_alpha-{a}_totalMC-{MC_iter}"
    suffix_0 = suffix + f"_ATTACK-1_MODE-0.txt"
    suffix_1 = suffix + f"_ATTACK-1_MODE-1.txt"
    # suffix_0 = f"m-{m}_e-{e}_bs-{bs}_alpha-{a}_totalMC-100" + f"_MODE-0.txt"
    # suffix_1 = f"m-{m}_e-{e}_bs-{bs}_alpha-{a}_totalMC-100" + f"_MODE-1.txt"
    suffix_2 = suffix + f"_ATTACK-1_MODE-2.txt"

    data = json.load(open(prefix + suffix_2))  # group testing

    # extract simulation data
    group_acc = np.array(data["group_acc"])
    acc = np.array(data["accuracy"])
    syndrome = np.array(data["syndrome"])
    DEC = np.array(data["DEC"]).astype(bool)
    syndrome = np.array(data["syndrome"])
    P_MD = np.array(data["P_MD"])
    P_FA = np.array(data["P_FA"])
    threshold_vec = np.array(data["threshold_vec"])
    n_clients = data["client_number"]
    n_malicious = data["n_malicious"]
    comm_rounds = data["comm_round"]
    malicious_clients = np.array(data["malicious_clients"]).astype(bool)

    try:
        data_no_defence = json.load(open(prefix0 + suffix_0))  # no defence
        acc_no_defence = np.array(data_no_defence["accuracy"]).squeeze()
    except:
        acc_no_defence = np.zeros((1, acc.shape[2]))

    try:
        data_oracle = json.load(open(prefix1 + suffix_1))  # oracle
        acc_oracle = np.array(data_oracle["accuracy"]).squeeze()
    except:
        acc_oracle = np.zeros((1, acc.shape[2]))

    plot_prefix = f"{d.upper()} with {MC_iter} MC iterations for {m} malicious clients"
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
    ax[0, 0].set_xlabel("Commuication rounds")
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 0].xaxis.grid(True)
    ax[0, 0].yaxis.grid(True)
    ax[0, 0].legend(str_legend)

    # FIG 2:
    # P_FA vs 1-P_MD
    ax[0, 1].plot(P_FA, 1 - P_MD, "*-")
    ax[0, 1].set_title("ROC")
    ax[0, 1].set_xlabel("P_FA")
    ax[0, 1].set_ylabel("1-P_MD")
    ax[0, 1].xaxis.grid(True)
    ax[0, 1].yaxis.grid(True)

    # FIG 3:
    # average accuracy wrt threshold after comm rounds
    final_acc = average_acc[:, -1]
    # find out how many malign/benign users are included
    benign_users_included = (1 - P_FA) * (n_clients - n_malicious)
    malign_users_included = P_MD * n_malicious
    ax[0, 2].bar(
        threshold_vec + 0.015,
        benign_users_included,
        width=0.02,
    )
    ax[0, 2].bar(threshold_vec - 0.015, malign_users_included, width=0.02)
    ax[0, 2].set_title(
        f"Average number of benign/malicious users included after group test @ {comm_rounds} rounds"
    )
    ax[0, 2].set_xlabel("GL Threshold")
    ax[0, 2].xaxis.grid(True)
    ax[0, 2].yaxis.grid(True)
    ax[0, 2].legend(["benign", "malicious"])

    # FIG
    ax[1, 0].plot(threshold_vec, final_acc)
    ax[1, 0].plot(
        threshold_vec, np.ones(threshold_vec.shape) * np.mean(acc_oracle[:, -1]), "--b"
    )
    ax[1, 0].plot(
        threshold_vec,
        np.ones(threshold_vec.shape) * np.mean(acc_no_defence[:, -1]),
        "--r",
    )
    ax[1, 0].set_title(f"Average accuracy @ {comm_rounds} rounds")
    ax[1, 0].set_xlabel("GL Threshold")
    ax[1, 0].xaxis.grid(True)
    ax[1, 0].yaxis.grid(True)
    ax[1, 0].legend(["group test", "oracle", "no defence"])

    # FIG 5:
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
    ax[1, 1].set_xlabel("GL Threshold")
    ax[1, 1].set_ylabel("Accuracy")
    ax[1, 1].xaxis.grid(True)
    ax[1, 1].yaxis.grid(True)
    ax[1, 1].legend(["group test", "oracle", "no defence"])

    # FIG 6:
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    str_legend = []
    for i in range(len(threshold_vec)):
        for j in range(10):  # MC_iter
            test = syndrome[i, j, :]
            u = np.unique(test).astype(int).tolist()
            for k in u:
                x = np.ones(group_acc[i, j, test == k].shape) * j
                if str(k) not in str_legend:
                    str_legend.append(str(k))
                    ax[1, 2].plot(
                        x,
                        group_acc[i, j, test == k],
                        "*",
                        color=color_cycle[k],
                        label=str(k),
                    )
                else:
                    ax[1, 2].plot(
                        x, group_acc[i, j, test == k], "*", color=color_cycle[k]
                    )
    ax[1, 2].legend()
    handles, labels = ax[1, 2].get_legend_handles_labels()
    order = np.argsort(np.array(str_legend))
    ax[1, 2].legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    ax[1, 2].set_title("Group accuracies")
    ax[1, 2].set_xlabel("MC iteration")
    ax[1, 2].set_ylabel("Accuracy")
    plt.show()
