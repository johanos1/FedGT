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

import numpy as np
from math import log

import ctypes
import matplotlib.pyplot as plt

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
    args.lr = 0.01
    args.wd = 0.0001
    args.epochs = 2
    args.comm_round = 2
    args.pretrained = True
    args.client_sample = 1.0
    args.thread_number = 5
    args.val_size = 3000
    args.method = "fedavg"  # fedavg, fedsgd
    args.data_dir = (
        "data/cifar10"  # data/cifar100, data/cifar10, data/mnist, data/fashionmnist
    )
    args.partition_method = "homo"  # homo, hetero
    args.partition_alpha = 0.1  # in (0,1]
    args.client_number = 15
    args.batch_size = 64

    # parameters related to the malicious users
    remove_detected_malicious_clients = False
    mali_number = 5

    results = np.zeros(4, 8)

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    set_random_seed()

    # -----------------------------------------
    #           Create attacks
    # -----------------------------------------
    malicious_clients = np.random.permutation(args.client_number)
    malicious_clients = malicious_clients[:mali_number].tolist()
    attacks = list_of_lists = [[] for i in range(args.client_number)]
    for client in range(args.client_number):
        if client in malicious_clients:
            label_flips = [(2, 5), (1, 7)]
            attacks[client].append((flip_label, label_flips))
            attacks[client].append((random_labels,))
            attacks[client].append((permute_labels,))

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
    args.partition_method = "homo"  # homo, hetero
    args.partition_alpha = 0.1  # in (0,1]
    args.client_number = 15

    lib = ctypes.cdll.LoadLibrary("./src/C_code/BCJR_4_python.so")
    fun = lib.BCJR
    fun.restype = None
    p_ui8_c = ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")
    p_d_c = ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")
    fun.argtypes = [
        p_ui8_c,
        p_d_c,
        p_ui8_c,
        p_d_c,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        p_d_c,
        p_ui8_c,
    ]

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
    #          Setup Group Testing
    # -----------------------------------------
    # Group testing parameters
    if args.client_number == 15:
    # fmt: off
        parity_check_matrix = np.array(
            [
                [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
            ],
            dtype=np.uint8,
        )
    elif args.client_number == 31:
        parity_check_matrix = np.array(
            [
                [
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                ],
            ],
            dtype=np.uint8,
        )
       # fmt: on

    number_tests = parity_check_matrix.shape[0]
    total_MC_it = 10
    # threshold_vec = np.arange(0, 0.6, 0.25).tolist()
    threshold_vec = np.arange(0.75, 0.8, 0.25).tolist()
    average_acc = np.zeros(len(threshold_vec))
    for indeks_group, threshold_dec in enumerate(threshold_vec):
        logging.info("Starting with threshold_dec : {}".format(threshold_dec))
        for monte_carlo_iterr in range(total_MC_it):
            set_random_seed(int(time.time()))
            # set_random_seed()
            # Create attacks
            mali_number = 5
            malicious_clients = np.random.permutation(args.client_number)
            malicious_clients = malicious_clients[:mali_number].tolist()
            defective = np.zeros((1, args.client_number), dtype=np.uint8)
            defective[:, malicious_clients] = 1
            attacks = list_of_lists = [[] for i in range(args.client_number)]
            for client in range(args.client_number):
                if client in malicious_clients:
                    # label_flips = [(1, 7), (3, 9)]
                    # label_flips = [(0, 1)]
                    # attacks[client].append((flip_label, label_flips))
                    # attacks[client].append((random_labels,))
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
            min_acc = 0.815
            threshold_from_max_acc = 0.99

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

            LLRO = np.empty((1, args.client_number), dtype=np.double)
            prevalence = mali_number / args.client_number
            LLRi = log((1 - prevalence) / prevalence) * np.ones(
                (1, args.client_number), dtype=np.double
            )
            ChannelMatrix = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.double)
            DEC = np.empty((1, args.client_number), dtype=np.uint8)

            # init nodes
            client_info = Queue()
            for i in range(args.thread_number):
                client_info.put((client_dict[i], args))

            syndrome = np.matmul(defective, parity_check_matrix.transpose())
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

                    # Test groups of clients on server validation set
                    if r == 0:
                        acc = np.zeros(number_tests)
                        f1 = []
                        for i in range(number_tests):
                            # np.where gives a tuple where first entry is the list we want
                            client_idxs = np.where(parity_check_matrix[i, :] == 1)[
                                0
                            ].tolist()
                            group = []
                            for idx in client_idxs:
                                group.append(client_outputs[idx])

                            # aggregation returns a list so pick the (only) item
                            model = server.aggregate_models(group, update_server=False)[
                                0
                            ]
                            # note, aside from accuracy, we have access to precision, recall, and f1 score for each class
                            (
                                acc[i],
                                class_precision,
                                class_recall,
                                class_f1,
                            ) = server.evaluate(test_data=False, eval_model=model)

                        max_acc = acc.max()
                        if max_acc < min_acc:
                            tests = np.ones((1, number_tests), dtype=np.uint8)
                        else:
                            tests = np.zeros((1, number_tests), dtype=np.uint8)
                            tests[:, acc < threshold_from_max_acc * max_acc] = 1
                        fun(
                            parity_check_matrix,
                            LLRi,
                            tests,
                            ChannelMatrix,
                            threshold_dec,
                            args.client_number,
                            number_tests,
                            LLRO,
                            DEC,
                        )
                        if (
                            np.sum(DEC) == DEC.shape[1]
                        ):  # , "All are classified as malicious"
                            all_class_malicious = True
                            break
                        # else:
                        #    aggregated_outputs_tested = [
                        #        client_outputs[kkkk]
                        #        for kkkk in range(args.client_number)
                        #        if DEC[:, kkkk] == 0
                        #    ]

                    # aggregate
                    # server_outputs = server.run(client_outputs)
                    aggregated_outputs_tested = [
                        client_outputs[kkkk]
                        for kkkk in range(args.client_number)
                        if DEC[:, kkkk] == 0
                    ]
                    server_outputs = server.run(aggregated_outputs_tested)
                    round_end = time.time()
                    logging.info(
                        "Round {} Time: {}s".format(r, round_end - round_start)
                    )
                if all_class_malicious == False:
                    model = server.aggregate_models(
                        aggregated_outputs_tested, update_server=False
                    )[0]
                    overall_acc, _, _, _ = server.evaluate(
                        test_data=True, eval_model=model
                    )
                else:
                    overall_acc = 1 / class_num
            average_acc[indeks_group] = average_acc[indeks_group] + overall_acc
        average_acc[indeks_group] = average_acc[indeks_group] / total_MC_it
        print(average_acc)
    np.savetxt("foo.csv", average_acc, delimiter=",")

