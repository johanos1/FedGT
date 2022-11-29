import torch
import numpy as np
import random
from dotmap import DotMap

import data_preprocessing.data_loader as dl
from models.resnet import resnet56, resnet18
from models.logistic_regression import logistic_regression
import logging
import os
from collections import defaultdict
import time
from math import floor, log

# from ctypes import * # Marvin: I changed this, maybe it crashes not much
import ctypes
from data_preprocessing.data_poisoning import flip_label, random_labels

# Calling C functions with numpy inputs
from numpy.ctypeslib import ndpointer

# methods
import methods.fedavg as fedavg
import methods.fedsgd as fedsgd

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

    # set_random_seed(2)
    set_random_seed(floor(time.time()))

    # # Parameters for calling BCJR C code
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
    min_accuracy = 0.5
    threshold_from_max_accuracy = 0.9
    ChannelMatrix = np.array(
        [[0.95, 0.05], [0.05, 0.95]], dtype=np.double
    )  # mismatched, needs to be estimated
    threshold_dec = 0.598

    # Set parameters
    args = DotMap()
    args.method = "fedavg"  # fedavg, fedsgd
    args.data_dir = (
        "data/mnist"  # data/cifar100, data/cifar10, data/mnist, data/fashionmnist
    )
    args.partition_method = "homo"  # homo, hetero
    args.partition_alpha = 0.1  # in (0,1]
    args.client_number = 15
    args.batch_size = 32
    args.lr = 0.05
    args.wd = 0.0001
    args.epochs = 3
    args.comm_round = 1
    args.pretrained = False
    args.client_sample = 1.0
    args.thread_number = 1
    args.val_size = 3000

    # Create attacks
    # malicious_clients = [0, 1, 2, 4, 5]
    LLRO = np.empty((1, args.client_number), dtype=np.double)
    DEC = np.empty((1, args.client_number), dtype=np.uint8)
    mali_number = 3
    prevalence = mali_number / args.client_number
    LLRi = log((1 - prevalence) / prevalence) * np.ones(
        (args.client_number), dtype=np.double
    )
    malicious_clients = np.random.permutation(args.client_number)
    malicious_clients = malicious_clients[:mali_number]
    defective_vector = np.zeros(args.client_number, dtype="uint8")
    defective_vector[malicious_clients] = 1
    attacks = list_of_lists = [[] for i in range(args.client_number)]
    for client in range(args.client_number):
        if client in malicious_clients:
            label_flips = [(1, 7), (2, 3)]
            attacks[client].append((flip_label, label_flips))
            # attacks[client].append((random_labels,))

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

    mapping_dict = defaultdict(list)
    for round in range(args.comm_round):
        num_clients = args.client_number
        for i, c in enumerate(range(num_clients)):
            mapping_dict[c].append([i])

    client_dict = [
        {
            "train_data": train_data_local_dict,
            "device": "cuda:{}".format(1) if torch.cuda.is_available() else "cpu",
            "client_map": mapping_dict[i],
            "model_type": Model,
            "num_classes": class_num,
        }
        for i in range(args.client_number)
    ]

    # init clients
    clients = []
    for c in client_dict:
        clients.append(Client(c, args))

    # Group testing parameters
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
        dtype="uint8",
    )
    # parity_check_matrix = np.eye(args.client_number, dtype="uint8")
    number_tests = parity_check_matrix.shape[0]
    syndrome = np.matmul(defective_vector, np.transpose(parity_check_matrix))
    # assert (
    #     parity_check_matrix.shape[1] == args.client_number
    # ), "Problem with size of parity check matrix!"

    # ----------------------------------------
    # Prepare Server info
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

    # ----------------------------------------
    # Run learning loop
    for r in range(args.comm_round):
        round_start = time.time()

        # train locally
        client_outputs = []
        for i, c in enumerate(clients):
            client_outputs.append(c.run(server_outputs[0]))

        client_outputs = [c for sublist in client_outputs for c in sublist]

        # Test groups of clients on server validation set
        acc = np.zeros(number_tests)
        f1 = []

        for i in range(number_tests):
            # np.where gives a tuple where first entry is the list we want
            client_idxs = np.where(parity_check_matrix[i, :] == 1)[0].tolist()
            group = []
            for idx in client_idxs:
                #    group.append(client_outputs[idx - 1])
                group.append(client_outputs[idx])

            # aggregation returns a list so pick the (only) item
            model = server.aggregate_models(group, update_server=False)[0]
            # note, aside from accuracy, we have access to precision, recall, and f1 score for each class
            acc[i], class_precision, class_recall, class_f1 = server.evaluate(
                test_data=False, eval_model=model
            )
        if r == 0:
            max_acc = acc.max()
            tests = np.zeros((number_tests), dtype="uint8")
            if max_acc < min_accuracy:
                tests = np.ones((number_tests), dtype="uint8")
            else:
                tests[acc < threshold_from_max_accuracy * max_acc] = 1
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

        # aggregate
        server_outputs = server.run(client_outputs)
        round_end = time.time()
        logging.info("Round {} Time: {}s".format(r, round_end - round_start))
        logging.info("-------------------------------------------------------")
