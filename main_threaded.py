import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False # prevent cuda from optimizing convolutional methods

import toml
import numpy as np
import random
from dotmap import DotMap
from torch.multiprocessing import set_start_method, Queue
import logging
from collections import defaultdict
import time
import json
from itertools import product
# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic#:~:text=As%20of%20Python,answer%20by%20jfs
from concurrent.futures import ProcessPoolExecutor as Pool

# methods
import data_preprocessing.data_loader as dl
import methods.fedavg as fedavg
from models.logistic_regression import logistic_regression
from models.eff_net import efficient_net


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

# 
def allocate_clients_to_threads(n_rounds, n_threads, n_clients):
    mapping_dict = defaultdict(list)
    for round in range(n_rounds):

        num_clients = n_clients
        if num_clients > 0:
            idxs = [[] for i in range(n_threads)]
            remaining_clients = num_clients
            thread_idx = 0
            client_idx = 0
            while remaining_clients > 0:
                idxs[thread_idx].extend([client_idx])

                remaining_clients -= 1
                thread_idx = (thread_idx + 1) % n_threads
                client_idx += 1
            for c, l in enumerate(idxs):
                mapping_dict[c].append(l)
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
    # torch.use_deterministic_algorithms(True) # only use deterministic algorithms

if __name__ == "__main__":

    #cfg_path = "./cfg_files/cfg_mnist.toml"
    cfg_path = "./cfg_files/cfg_cifar.toml"
    
    with open(cfg_path, "r") as file:
        cfg = DotMap(toml.load(file))
    
    sim_result = {}

    sim_params = list(product(cfg.Data.alpha_list, cfg.ML.epochs_list, cfg.ML.batch_size_list))

    for (alpha, epochs, batch_size) in sim_params:        

        # prepare to store results
        sim_result["epochs"] = epochs
        sim_result["val_size"] = cfg.Data.val_size
        sim_result["batch_size"] = batch_size
        sim_result["alpha"] = alpha
        sim_result["data_dir"] = cfg.Data.data_dir
        sim_result["lr"] = cfg.ML.lr
        sim_result["wd"] = cfg.ML.wd
        sim_result["momentum"] = cfg.ML.momentum
        sim_result["comm_round"] = cfg.ML.communication_rounds
        sim_result["client_number"] = cfg.Sim.n_clients
        sim_result["total_MC_it"] = cfg.Sim.total_MC_it
        sim_result["accuracy"] = np.zeros((cfg.Sim.total_MC_it, cfg.ML.communication_rounds))
        sim_result["cf_matrix"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds,cfg.Data.n_classes,cfg.Data.n_classes,))

        try:
            set_start_method("spawn")
        except RuntimeError:
            pass

        accuracy = np.zeros(( cfg.Sim.total_MC_it, cfg.ML.communication_rounds))

        logging.info("Starting Simulation")
        for monte_carlo_iterr in range(cfg.Sim.total_MC_it):
            logging.info("Monte Carlo Iteration: {}".format(monte_carlo_iterr))
            set_random_seed(monte_carlo_iterr) 
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
                cfg.Data.data_dir,
                cfg.Data.partition_method,
                alpha,
                cfg.Sim.n_clients,
                batch_size,
                cfg.Data.val_size,
            )

            # -----------------------------------------
            #         Choose Model and FL protocol
            # -----------------------------------------
            if "cifar" in cfg.Data.data_dir:
                Model = efficient_net
            elif "mnist" in cfg.Data.data_dir:
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
                "num_classes": cfg.Data.n_classes,
            }
            server_args = DotMap()
            server_args.n_threads = cfg.Sim.n_threads
            server_args.aggregation = "FedADAM" # overwritten!!!
            #server_args.aggregation = "FedAdagrad" # overwritten!!!
            #server_args.aggregation = "FedYogi" # overwritten!!!
            print(f"Alpha is equal to {alpha} and aggregation method is {server_args.aggregation}")
            server = Server(server_dict, server_args)
            server_outputs = server.start() # get global model to start from

            # -----------------------------------------
            #               Setup Clients
            # -----------------------------------------
            mapping_dict = allocate_clients_to_threads(
                cfg.ML.communication_rounds, cfg.Sim.n_threads, cfg.Sim.n_clients
            )
            client_dict = [
                {
                    "train_data": [
                        train_data_local_dict[j] for j in mapping_dict[i][0]
                    ],
                    "device": "cuda:{}".format(i % torch.cuda.device_count())
                    if torch.cuda.is_available()
                    else "cpu",
                    "client_map": mapping_dict[i],
                    "model_type": Model,
                    "num_classes": cfg.Data.n_classes,
                }
                for i in range(cfg.Sim.n_threads)
            ]
            # init nodes
            client_args = DotMap()
            client_args.epochs = epochs
            client_args.batch_size = batch_size
            client_args.lr = cfg.ML.lr
            client_args.momentum = cfg.ML.momentum
            client_args.wd = cfg.ML.wd
            
            client_info = Queue()
            for i in range(cfg.Sim.n_threads):
                client_info.put((client_dict[i], client_args))

            # -----------------------------------------
            #            Main Loop
            # -----------------------------------------
            # each thread will create a client object containing the client information
            acc = np.zeros((1, cfg.ML.communication_rounds))

            with Pool(
                max_workers=cfg.Sim.n_threads,
                initializer=init_process,
                initargs=(client_info, Client),
            ) as pool:
                for r in range(0, cfg.ML.communication_rounds):
                    round_start = time.time()
                    
                    # -----------------------------------------
                    #         Perform local training
                    # -----------------------------------------
                    client_outputs = pool.map(run_clients, server_outputs)
                    client_outputs = [c for sublist in client_outputs for c in sublist]
                    client_outputs.sort(key=lambda tup: tup["client_index"])

                    # -----------------------------------------
                    #               Aggregation
                    # -----------------------------------------
                    server_outputs, acc[0, r], cf_matrix = server.run(client_outputs)
                    

                    sim_result["cf_matrix"][monte_carlo_iterr, r, :, :] = cf_matrix # store confusion matrix

                    round_end = time.time()
                    
                    logging.info(f"MC-Iteration: {monte_carlo_iterr} --- Round {r} ---  Time: {round_end - round_start} --- Accuracy: {acc[0,r]}")
                        
                sim_result["accuracy"][monte_carlo_iterr, :] = acc

        if "mnist" in cfg.Data.data_dir:
            prefix = "./results/MNIST_"
        elif "cifar" in cfg.Data.data_dir:
            prefix = "./results/CIFAR10_"
        suffix = f"_e-{epochs}_bs-{batch_size}_alpha-{alpha}_totalMC-{cfg.Sim.total_MC_it}.txt"
        sim_title = prefix + suffix

        with open(sim_title, "w") as convert_file:
            convert_file.write(json.dumps(sim_result))


