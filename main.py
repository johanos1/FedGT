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

    # Test calling c file
    # so_file = "./src/C_code/my_functions.so"
    # my_functions = ctypes.CDLL(so_file)
    # print(my_functions.square(10))

    set_random_seed()

    # Parameters for calling BCJR C code
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

    # Set parameters
    args = DotMap()
    args.method = "fedsgd"  # fedavg, fedsgd
    args.data_dir = (
        "data/mnist"  # data/cifar100, data/cifar10, data/mnist, data/fashionmnist
    )
    args.partition_method = "homo"  # homo, hetero
    args.partition_alpha = 0.1  # in (0,1]
    args.client_number = 15
    args.batch_size = 100
    args.lr = 0.01
    args.wd = 0.0001
    args.epochs = 1
    args.comm_round = 5
    args.pretrained = False
    args.client_sample = 1.0
    args.thread_number = 1
    args.val_size = 3000

    # define attacks
    malicious_clients = [0, 1, 2]

    attacks = list_of_lists = [[] for i in range(args.client_number)]
    for client in range(args.client_number):
        if client in malicious_clients:
            # label_flips = [(0, 1), (3, 2)]
            # attacks[client].append((flip_label, label_flips))
            attacks[client].append((random_labels,))

    # get data
    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
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
            "test_data": test_data_local_dict,
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
        ]
    )
    number_tests = parity_check_matrix.shape[0]
    assert (
        parity_check_matrix.shape[1] == args.client_number
    ), "Problem with size of parity check matrix!"

    # ----------------------------------------
    # Prepare Server info
    server_dict = {
        "train_data": train_data_global,
        "test_data": test_data_global,
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

        # This is where the identification of malicious nodes should go
        # TODO: write testing
        # Testing part
        accuracies = []
        for test in range(number_tests):
            testing_rule = parity_check_matrix[test, :]
            testing_clients = [
                client_outputs[i]
                for i in range(args.client_number)
                if testing_rule[i] == 1
            ]
            server_outputs = server.run(testing_clients)
            accuracies.append(server.acc)

        # aggregate
        server_outputs = server.run(client_outputs)
        round_end = time.time()
        logging.info("Round {} Time: {}s".format(r, round_end - round_start))
        logging.info("-------------------------------------------------------")
