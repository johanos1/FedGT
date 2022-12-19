import torch
import numpy as np
import random
from dotmap import DotMap

import data_preprocessing.data_loader as dl
from models.resnet import resnet56, resnet18


from models.logistic_regression import logistic_regression
from defence.group_test import Group_Test

from torch.multiprocessing import set_start_method, Queue
import logging

import os
from collections import defaultdict
import time

import numpy as np

import matplotlib.pyplot as plt
from data_preprocessing.data_poisoning import flip_label, random_labels, permute_labels

import json
from itertools import product

# methods
import methods.fedavg as fedavg
import methods.fedsgd as fedsgd

# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic#:~:text=As%20of%20Python,answer%20by%20jfs
from concurrent.futures import ProcessPoolExecutor as Pool

# Helper Functions
def init_process(q, Client):
    set_random_seed()
    global client
    ci = q.get()
    client = Client(ci[0], ci[1])


def run_clients(received_info):
    try:
        return client.run(received_info)
    except KeyboardInterrupt:
        logging.info("exiting")
        return None


def allocate_clients_to_threads(args):
    mapping_dict = defaultdict(list)
    for round in range(args.comm_round):
        if args.client_sample < 1.0:
            num_clients = int(args.client_number * args.client_sample)
            client_list = random.sample(range(args.client_number), num_clients)
        else:
            num_clients = args.client_number
            client_list = list(range(num_clients))
        if num_clients % args.thread_number == 0 and num_clients > 0:
            clients_per_thread = int(num_clients / args.thread_number)
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t + clients_per_thread)]
                mapping_dict[c].append(idxs)
        else:
            raise ValueError("Sampled client number not divisible by number of threads")
    return mapping_dict


# Setup Functions
def set_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    args = DotMap()
    args.comm_round = 5
    args.pretrained = False
    args.client_sample = 1.0
    args.thread_number = 5
    args.val_size = 3000
    args.method = "fedavg"  # fedavg, fedsgd
    args.data_dir = (
        "data/mnist"  # data/cifar100, data/cifar10, data/mnist, data/fashionmnist
    )
    args.partition_method = "homo"  # homo, hetero
    args.client_number = 15

    if "cifar" in args.data_dir:
        args.lr = 0.05
        args.momentum = 0.9
        args.wd = 0.001
    else:
        args.lr = 0.01
        args.momentum = 0
        args.wd = 0

    total_MC_it = 10
    threshold_vec = np.arange(0.1, 0.8, 0.2).tolist()
    sim_result = {}

    # Set hyper parameters to sweep
    if args.partition_method == "homo":
        alpha_list = [np.inf]
    else:
        alpha_list = [10]

    epochs_list = [1]
    n_malicious_list = [5]
    batch_size_list = [64]

    sim_params = list(
        product(alpha_list, epochs_list, n_malicious_list, batch_size_list)
    )

    for (alpha, epochs, n_malicious, batch_size) in sim_params:

        # prepare to store results
        sim_result["epochs"] = epochs
        sim_result["batch_size"] = batch_size
        sim_result["alpha"] = alpha
        sim_result["n_malicious"] = n_malicious
        sim_result["data_dir"] = args.data_dir
        sim_result["lr"] = args.lr
        sim_result["wd"] = args.wd
        sim_result["momentum"] = args.momentum
        sim_result["comm_round"] = args.comm_round
        sim_result["client_number"] = args.client_number
        sim_result["total_MC_it"] = total_MC_it
        sim_result["threshold_vec"] = threshold_vec
        sim_result["group_acc"] = np.zeros((len(threshold_vec), total_MC_it, 8))
        sim_result["malicious_clients"] = np.zeros(
            (len(threshold_vec), total_MC_it, n_malicious)
        )
        sim_result["DEC"] = np.zeros(
            (len(threshold_vec), total_MC_it, args.client_number)
        )
        sim_result["syndrome"] = np.zeros((len(threshold_vec), total_MC_it, 8))
        sim_result["accuracy"] = np.zeros(
            (len(threshold_vec), total_MC_it, args.comm_round)
        )

        args.partition_alpha = alpha
        args.epochs = epochs
        args.batch_size = batch_size

        try:
            set_start_method("spawn")
        except RuntimeError:
            pass

        accuracy = np.zeros((len(threshold_vec), total_MC_it, args.comm_round))
        P_FA = np.zeros(len(threshold_vec))
        P_MD = np.zeros(len(threshold_vec))

        for thres_indx, threshold_dec in enumerate(threshold_vec):
            logging.info("Starting with threshold_dec : {}".format(threshold_dec))
            FA = 0
            MD = 0
            set_random_seed()  # all mc iterations should have same seed for each threshold value

            for monte_carlo_iterr in range(total_MC_it):

                # -----------------------------------------
                #           Create attacks
                # -----------------------------------------
                malicious_clients = np.random.permutation(args.client_number)
                malicious_clients = malicious_clients[:n_malicious].tolist()
                defective = np.zeros((1, args.client_number), dtype=np.uint8)
                defective[:, malicious_clients] = 1
                attacks = list_of_lists = [[] for i in range(args.client_number)]
                for client in range(args.client_number):
                    if client in malicious_clients:
                        # label_flips = [(1, 7), (3, 9)]
                        # label_flips = [(1, 7)]
                        # attacks[client].append((flip_label, label_flips))
                        # attacks[client].append((random_labels,))
                        attacks[client].append((permute_labels,))

                sim_result["malicious_clients"][
                    thres_indx, monte_carlo_iterr, :
                ] = defective
                # -----------------------------------------
                # Obtain dataset for server and the clients
                # -----------------------------------------
                (
                    val_data_num,
                    test_data_num,
                    server_val_dl,
                    server_test_dl,
                    data_local_num_dict,
                    train_data_local_dict,
                    class_num,
                ) = dl.load_partition_data(
                    args.data_dir,
                    args.partition_method,
                    args.partition_alpha,
                    args.client_number,
                    args.batch_size,
                    attacks,
                    args.val_size,
                )

                # -----------------------------------------
                #         Choose Model and FL protocol
                # -----------------------------------------
                if "cifar" in args.data_dir:
                    Model = resnet18
                elif "mnist" in args.data_dir:
                    Model = logistic_regression

                # Pick FL method
                Server = fedavg.Server
                Client = fedavg.Client

                # -----------------------------------------
                #               Setup Server
                # -----------------------------------------
                server_dict = {
                    "val_data": server_val_dl,
                    "test_data": server_test_dl,
                    "model_type": Model,
                    "num_classes": class_num,
                }

                # init server
                server_dict["save_path"] = "{}/logs/{}__{}_e{}_c{}".format(
                    os.getcwd(),
                    time.strftime("%Y%m%d_%H%M%S"),
                    args.method,
                    args.epochs,
                    args.client_number,
                )
                if not os.path.exists(server_dict["save_path"]):
                    os.makedirs(server_dict["save_path"])
                server = Server(server_dict, args)
                # get global model to start from
                server_outputs = server.start()

                # -----------------------------------------
                #               Setup Clients
                # -----------------------------------------
                mapping_dict = allocate_clients_to_threads(args)
                client_dict = [
                    {
                        "train_data": train_data_local_dict,
                        "device": "cuda:{}".format(i % torch.cuda.device_count())
                        if torch.cuda.is_available()
                        else "cpu",
                        "client_map": mapping_dict[i],
                        "model_type": Model,
                        "num_classes": class_num,
                    }
                    for i in range(args.thread_number)
                ]
                # init nodes
                client_info = Queue()
                for i in range(args.thread_number):
                    client_info.put((client_dict[i], args))

                # -----------------------------------------
                #          Make Group Test Object
                # -----------------------------------------
                prevalence = n_malicious / args.client_number
                gt = Group_Test(
                    args.client_number,
                    prevalence,
                    threshold_dec,
                    min_acc=0.815,
                    threshold_from_max_acc=0.99,
                    P_FA_test=0.05,
                    P_MD_test=0.05,
                )

                syndrome = np.matmul(defective, gt.parity_check_matrix.transpose())

                # -----------------------------------------
                #            Main Loop
                # -----------------------------------------
                # each thread will create a client object containing the client information
                acc = np.zeros((1, args.comm_round))
                all_class_malicious = False
                with Pool(
                    max_workers=args.thread_number,
                    initializer=init_process,
                    initargs=(client_info, Client),
                ) as pool:
                    for r in range(args.comm_round):
                        logging.info(
                            f"************** Round: {r}, MC-Iteration: {monte_carlo_iterr}  ***************"
                        )
                        round_start = time.time()

                        # -----------------------------------------
                        #         Perform local training
                        # -----------------------------------------
                        client_outputs = pool.map(run_clients, server_outputs)
                        client_outputs = [
                            c for sublist in client_outputs for c in sublist
                        ]
                        client_outputs.sort(key=lambda tup: tup["client_index"])

                        # -----------------------------------------
                        #           Group Testing
                        # -----------------------------------------
                        if r == 0:
                            group_accuracies = gt.get_group_accuracies(
                                client_outputs, server
                            )
                            DEC = gt.perform_group_test(group_accuracies)
                            MD = MD + np.sum(gt.DEC[defective == 1] == 0)
                            FA = FA + np.sum(gt.DEC[defective == 0] == 1)
                            if (
                                np.sum(DEC) == DEC.shape[1]
                            ):  # , "All are classified as malicious"
                                all_class_malicious = True

                            sim_result["group_acc"][
                                thres_indx, monte_carlo_iterr, :
                            ] = group_accuracies
                            sim_result["DEC"][thres_indx, monte_carlo_iterr, :] = DEC
                            sim_result["syndrome"][
                                thres_indx, monte_carlo_iterr
                            ] = syndrome[0]
                        # -----------------------------------------
                        #               Aggregation
                        # -----------------------------------------
                        # If all malicious, just use all
                        if all_class_malicious == True:
                            clients_to_aggregate = client_outputs
                        else:
                            clients_to_aggregate = [
                                client_outputs[client_idx]
                                for client_idx in range(args.client_number)
                                if DEC[:, client_idx] == 0
                            ]
                        server_outputs, acc[0, r] = server.run(clients_to_aggregate)
                        round_end = time.time()
                        logging.info(f"Round {r} Time: {round_end - round_start}")

                        logging.info(f"************* Acc = {acc[0,r]} **************")

                    accuracy[thres_indx, monte_carlo_iterr, :] = acc

            P_MD[thres_indx] = MD / (n_malicious * total_MC_it)
            P_FA[thres_indx] = FA / ((args.client_number - n_malicious) * total_MC_it)

        # make all nparrays JSON serializable
        sim_result["accuracy"] = accuracy.tolist()
        sim_result["P_MD"] = P_MD.tolist()
        sim_result["P_FA"] = P_FA.tolist()
        sim_result["group_acc"] = sim_result["group_acc"].tolist()
        sim_result["DEC"] = sim_result["DEC"].tolist()
        sim_result["syndrome"] = sim_result["syndrome"].tolist()
        sim_result["malicious_clients"] = sim_result["malicious_clients"].tolist()

        if "mnist" in args.data_dir:
            prefix = "./results/MNIST_"
        elif "cifar" in args.data_dir:
            prefix = "./results/CIFAR10_"
        suffix = f"m-{n_malicious}_e-{args.epochs}_bs-{args.batch_size}_alpha-{args.partition_alpha}-totalMC-{total_MC_it}.txt"
        sim_title = prefix + suffix

        with open(sim_title, "w") as convert_file:
            convert_file.write(json.dumps(sim_result))
