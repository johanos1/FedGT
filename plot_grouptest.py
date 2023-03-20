from matplotlib import pyplot as plt
import json
import numpy as np
from itertools import product


data = ["mnist"]  # , "cifar10"]
if data[0] == "cifar10":
    MC_iter = 10
    epochs_list = [5]
    n_malicious_list = [2]
    batch_size_list = [128]
    n_client_list = [15]
elif data[0] == "mnist":
    MC_iter = 10
    epochs_list = [1]
    n_malicious_list = [5]
    batch_size_list = [64]
    n_client_list = [15]

ATTACK = 0

if ATTACK == 0:
    att_str = "(permutation attack)"
elif ATTACK == 1:
    att_str = "(random permutation attack)"
elif ATTACK == 2:
    att_str = "(1->7 label flip attack)"
    src = 1
    target = 7

alpha_list = [0.5]  # [np.inf]
sim_params = list(product(data, alpha_list, epochs_list, n_malicious_list, n_client_list, batch_size_list))

for k, (d, a, e, m, n, bs) in enumerate(sim_params):

    fig, ax = plt.subplots(nrows=3, ncols=3)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
    # load file
    prefix = f"./results/{d.upper()}_"
    suffix = f"m-{m},{n}_e-{e}_bs-{bs}_alpha-{a}_totalMC-{MC_iter}"
    suffix_0 = suffix + f"_MODE-0_att-{ATTACK}.txt"
    suffix_1 = suffix + f"_MODE-1_att-{ATTACK}.txt"
    suffix_2 = suffix + f"_MODE-2_att-{ATTACK}.txt"

    data = json.load(open(prefix + suffix_2))  # group testing

    # extract simulation data
    group_acc = np.array(data["group_acc"])

    acc = np.array(data["accuracy"])
    syndrome = np.array(data["syndrome"])
    DEC = np.array(data["DEC"]).astype(bool)
    P_MD = np.array(data["P_MD"])
    P_FA = np.array(data["P_FA"])
    threshold_vec = np.array(data["threshold_vec"])
    n_clients = data["client_number"]
    n_malicious = data["n_malicious"]
    comm_rounds = data["comm_round"]
    malicious_clients = np.array(data["malicious_clients"]).astype(bool)

    try:
        data_no_defence = json.load(open(prefix + suffix_0))  # no defence
        acc_no_defence = np.array(data_no_defence["accuracy"]).squeeze()
        if "cf_matrix" in data_no_defence:
            cf_no_defence = np.array(data_no_defence["cf_matrix"])
    except:
        acc_no_defence = np.zeros((1, acc.shape[2]))

    try:
        data_oracle = json.load(open(prefix + suffix_1))  # oracle
        acc_oracle = np.array(data_oracle["accuracy"]).squeeze()
        if "cf_matrix" in data_oracle:
            cf_oracle = np.array(data_oracle["cf_matrix"])
    except:
        acc_oracle = np.zeros((1, acc.shape[2]))

    alpha_str = r"$\alpha$"
    plot_prefix = (
        f"{d.upper()}(" + alpha_str + f" = {a}) with {MC_iter} MC iterations for {n_malicious}/{n_clients} " + att_str
    )
    fig.suptitle(plot_prefix)
    # ------------------------------------------------------------
    # FIG 1:
    # average accuracy over communication rounds
    # acc is structured as: thresholds x MC iter x comm rounds
    # ------------------------------------------------------------
    average_acc = np.mean(acc, axis=1)
    for i, threshold in enumerate(threshold_vec):
        leg = str("{0:.2f}".format(threshold))
        ax[0, 0].plot(range(comm_rounds), average_acc[i, :], label=leg)
    ax[0, 0].plot(range(comm_rounds), np.mean(acc_oracle, axis=0), "--b", label="oracle")
    ax[0, 0].plot(range(comm_rounds), np.mean(acc_no_defence, axis=0), "--r", label="no defence")
    ax[0, 0].set_title("Average accuracy")
    ax[0, 0].set_xlabel("Commuication rounds")
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 0].xaxis.grid(True)
    ax[0, 0].yaxis.grid(True)

    # ------------------------------------------------------------
    # FIG 2:
    # average target accuracy over communication rounds
    # how many
    # ------------------------------------------------------------
    if ATTACK == 2:
        try:
            group_prec = np.array(data["group_prec"])
            group_rec = np.array(data["group_recall"])
            group_f1 = np.array(data["group_f1"])
        except:
            group_prec = np.zeros((group_acc.shape, 1))
            group_rec = np.zeros((group_acc.shape, 1))
            group_f1 = np.zeros((group_acc.shape, 1))

        cf_matrix = np.array(data["cf_matrix"])
        # cfmatrix: thresholds x mc iter x comm rounds x true label x predicted label
        tot_src = np.sum(cf_matrix[:, :, :, src, :], axis=3)[0, 0, 0]
        attack_success = cf_matrix[:, :, :, src, target]
        target_accuracy = attack_success / tot_src
        target_accuracy = np.mean(target_accuracy, axis=1)

        ta_nd = cf_no_defence[0, :, :, src, target] / tot_src
        ta_nd = np.mean(ta_nd, axis=0)
        ta_oracle = cf_oracle[0, :, :, src, target] / tot_src
        ta_oracle = np.mean(ta_oracle, axis=0)

        for i, threshold in enumerate(threshold_vec):
            leg = str("{0:.2f}".format(threshold))
            ax[0, 1].plot(range(comm_rounds), target_accuracy[i, :], label=leg)
        ax[0, 1].plot(range(comm_rounds), ta_oracle, "--b", label="oracle")
        ax[0, 1].plot(range(comm_rounds), ta_nd, "--r", label="no defence")

    ax[0, 1].set_title("Average target accuracy")
    ax[0, 1].set_xlabel("Commuication rounds")
    ax[0, 1].set_ylabel("Accuracy")
    ax[0, 1].xaxis.grid(True)
    ax[0, 1].yaxis.grid(True)
    # ------------------------------------------------------------
    # FIG 3:
    # ROC
    # ------------------------------------------------------------
    ax[0, 2].plot(P_FA, 1 - P_MD, "*-")
    ax[0, 2].set_title("ROC")
    ax[0, 2].set_xlabel("P_FA")
    ax[0, 2].set_ylabel("1-P_MD")
    ax[0, 2].xaxis.grid(True)
    ax[0, 2].yaxis.grid(True)

    # ------------------------------------------------------------
    # FIG 4:
    # average accuracy wrt threshold after comm rounds
    # ------------------------------------------------------------
    final_acc = average_acc[:, -1]
    ax[1, 0].plot(threshold_vec, final_acc, label="group test")
    ax[1, 0].plot(
        threshold_vec,
        np.ones(threshold_vec.shape) * np.mean(acc_oracle[:, -1]),
        "--b",
        label="oracle",
    )
    ax[1, 0].plot(
        threshold_vec,
        np.ones(threshold_vec.shape) * np.mean(acc_no_defence[:, -1]),
        "--r",
        label="no defence",
    )
    ax[1, 0].set_title(f"Average accuracy @ {comm_rounds} rounds")
    ax[1, 0].set_xlabel("GL Threshold")
    ax[1, 0].xaxis.grid(True)
    ax[1, 0].yaxis.grid(True)

    # ------------------------------------------------------------
    # FIG 5:
    # average target accuracy wrt threshold after comm rounds
    # ------------------------------------------------------------
    if ATTACK == 2:
        avg_target_accuracy = target_accuracy[:, -1]
        ax[1, 1].plot(threshold_vec, avg_target_accuracy, label="group test")
        ax[1, 1].plot(
            threshold_vec,
            np.ones(threshold_vec.shape) * np.mean(ta_oracle[-1]),
            "--b",
            label="oracle",
        )
        ax[1, 1].plot(
            threshold_vec,
            np.ones(threshold_vec.shape) * np.mean(ta_nd[-1]),
            "--r",
            label="no defence",
        )
        ax[1, 1].set_title(f"Average target accuracy @ {comm_rounds} rounds")
        ax[1, 1].set_xlabel("GL Threshold")
        ax[1, 1].set_ylabel("Accuracy")
        ax[1, 1].xaxis.grid(True)
        ax[1, 1].yaxis.grid(True)

    # ------------------------------------------------------------
    # FIG 6:
    # Number of benign/malicious users included after GT
    # ------------------------------------------------------------
    # find out how many malign/benign users are included
    benign_users_included = (1 - P_FA) * (n_clients - n_malicious)
    malign_users_included = P_MD * n_malicious
    ax[1, 2].bar(threshold_vec + 0.015, benign_users_included, width=0.02, label="benign")
    ax[1, 2].bar(threshold_vec - 0.015, malign_users_included, width=0.02, label="malicious")
    ax[1, 2].set_title(f"Avg nodes after group test @ {comm_rounds} rounds")
    ax[1, 2].set_xlabel("GL Threshold")
    ax[1, 2].xaxis.grid(True)
    ax[1, 2].yaxis.grid(True)

    # ------------------------------------------------------------
    # FIG 7:
    # Group accuracies going into GT
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    str_legend = []
    for i in range(len(threshold_vec)):
        for j in range(MC_iter):
            test = syndrome[i, j, :]
            u = np.unique(test).astype(int).tolist()
            for k in u:
                x = np.ones(group_acc[i, j, test == k].shape) * j
                if str(k) not in str_legend:
                    str_legend.append(str(k))
                    ax[2, 0].plot(
                        x,
                        group_acc[i, j, test == k],
                        "*",
                        color=color_cycle[k],
                        label=str(k),
                    )
                else:
                    ax[2, 0].plot(x, group_acc[i, j, test == k], "*", color=color_cycle[k])

    ax[2, 0].set_title("Group accuracies")
    ax[2, 0].set_xlabel("MC iteration")
    ax[2, 0].set_ylabel("Accuracy")
    # FIG 8:
    # Class precision before GT
    # thresholds x mc iter x syndrome x class label

    # FIG 9:
    # Class recall before GT
    # ------------------------------------------------------------
    if ATTACK == 2:
        num_classes = group_prec.shape[3]
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        str_legend = []
        for i in range(1):
            for j in range(MC_iter):
                test = syndrome[i, j, :]
                u = np.unique(test).astype(int).tolist()
                for k in u:
                    x = np.ones(group_acc[i, j, test == k].shape) * j
                    if str(k) not in str_legend:
                        str_legend.append(str(k))
                        ax[2, 1].plot(
                            x,
                            group_prec[i, j, test == k, target].squeeze(),
                            "*",
                            color=color_cycle[k],
                            label=str(k),
                        )
                        ax[2, 2].plot(
                            x,
                            group_rec[i, j, test == k, src].squeeze(),
                            "*",
                            color=color_cycle[k],
                            label=str(k),
                        )
                    else:
                        ax[2, 1].plot(
                            x,
                            group_prec[i, j, test == k, target].squeeze(),
                            "*",
                            color=color_cycle[k],
                        )
                        ax[2, 2].plot(
                            x,
                            group_rec[i, j, test == k, src].squeeze(),
                            "*",
                            color=color_cycle[k],
                        )

        ax[2, 1].set_title(f"Target precision ({target})")
        ax[2, 1].set_xlabel("MC iteration")
        ax[2, 1].set_ylabel("Precision")

        ax[2, 2].set_title(f"Source recall ({src})")
        ax[2, 2].set_xlabel("MC iteration")
        ax[2, 2].set_ylabel("Recall")

        ax[2, 0].legend()
        handles, labels = ax[2, 0].get_legend_handles_labels()
        order = np.argsort(np.array(str_legend))
        ax[2, 0].legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            bbox_to_anchor=(-0.15, 1),
        )

    # fig.tight_layout()
    # add all legends
    ax[0, 0].legend(bbox_to_anchor=(-0.15, 1))
    ax[1, 0].legend()
    ax[1, 1].legend()
    ax[1, 2].legend()

    plt.show()
