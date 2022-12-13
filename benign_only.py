import torch
import numpy as np
import random
from dotmap import DotMap

import data_preprocessing.data_loader as dl
from models.resnet import resnet56, resnet18
from models.logistic_regression import logistic_regression
from torch.multiprocessing import set_start_method, Queue
import logging
import os
from collections import defaultdict
import time
from math import log

import ctypes

# Calling C functions with numpy inputs
from numpy.ctypeslib import ndpointer

from data_preprocessing.data_poisoning import flip_label, random_labels, permute_labels

# methods
import methods.fedavg as fedavg
import methods.fedsgd as fedsgd

# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic#:~:text=As%20of%20Python,answer%20by%20jfs
from concurrent.futures import ProcessPoolExecutor as Pool

# Helper Functions
def init_process(q, Client):
    # set_random_seed()
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

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    # get arguments
    total_MC_it = 100
    rounds_tot = 5
    args = DotMap()
    args.lr = 0.01
    args.wd = 0.0001
    args.batch_size = 64
    args.epochs = 1
    args.comm_round = rounds_tot
    args.pretrained = False
    args.client_sample = 1.0
    args.thread_number = 5
    args.val_size = 3000
    args.method = "fedavg"  # fedavg, fedsgd
    args.data_dir = (
        "data/mnist"  # data/cifar100, data/cifar10, data/mnist, data/fashionmnist
    )
    args.partition_method = "homo"  # homo, hetero
    args.partition_alpha = 0.1  # in (0,1]
    args.client_number = 15

    # Model = resnet56 if 'cifar' in args.data_dir else resnet18
    if "cifar" in args.data_dir:
        Model = resnet18
    elif "mnist" in args.data_dir:
        Model = logistic_regression

    # init method and model type
    if args.method == "fedavg":
        Client = fedavg.Client
        Server = fedavg.Server

    elif args.method == "fedsgd":
        Client = fedsgd.Client
        Server = fedsgd.Server

    # average_acc = np.zeros(len(threshold_vec))
    average_acc = 0
    sidja = int(time.time())
    for monte_carlo_iterr in range(total_MC_it):
        set_random_seed(sidja)
        # Create attacks
        mali_number = 5
        malicious_clients = np.random.permutation(args.client_number)
        malicious_clients = malicious_clients[:mali_number].tolist()
        attacks = list_of_lists = [[] for i in range(args.client_number)]
        for client in range(args.client_number):
            if client in malicious_clients:
                attacks[client].append((permute_labels,))
        # Obtain dataset for server and the clients
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
        mapping_dict = allocate_clients_to_threads(args)
        # init method and model type
        if args.method == "fedavg":
            Server = fedavg.Server
            Client = fedavg.Client
            # Model = resnet56 if 'cifar' in args.data_dir else resnet18
            Model = logistic_regression
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
        server_outputs = server.start()

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

        # each thread will create a client object containing the client information
        with Pool(
            max_workers=args.thread_number,
            initializer=init_process,
            initargs=(client_info, Client),
        ) as pool:
            for r in range(args.comm_round):
                all_class_malicious = False
                logging.info(
                    "************** Round: {}, MC-Iteration: {}  ***************".format(
                        r, monte_carlo_iterr
                    )
                )
                round_start = time.time()
                client_outputs = pool.map(run_clients, server_outputs)
                client_outputs = [c for sublist in client_outputs for c in sublist]
                client_outputs.sort(
                    key=lambda tup: tup["client_index"]
                )  # Added this ....
                aggregated_outputs_tested = [
                    client_outputs[kkkk]
                    for kkkk in range(args.client_number)
                    if kkkk not in malicious_clients
                ]
                server_outputs = server.run(aggregated_outputs_tested)
                round_end = time.time()
                logging.info("Round {} Time: {}s".format(r, round_end - round_start))
            model = server.aggregate_models(
                aggregated_outputs_tested, update_server=False
            )[0]
            overall_acc, _, _, _ = server.evaluate(test_data=True, eval_model=model)
            average_acc = average_acc + overall_acc
    average_acc = average_acc / total_MC_it
    print(average_acc)
    np.savetxt("benign_only_acc.csv", np.array([average_acc]), delimiter=",")
