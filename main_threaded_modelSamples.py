import toml
import random
import logging
from collections import defaultdict
import math
import time
import json
import pickle
from itertools import product
# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic#:~:text=As%20of%20Python,answer%20by%20jfs
from concurrent.futures import ProcessPoolExecutor as Pool

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False # prevent cuda from optimizing convolutional methods
from dotmap import DotMap
from torch.multiprocessing import set_start_method, Queue
import numpy as np

# methods
import data_preprocessing.data_loader as dl
import methods.fedavg as fedavg
from models.logistic_regression import logistic_regression
from models.eff_net import efficient_net

# a = multiprocessing.cpu_count()
# print(a) 

# Helper Functions
def init_process(q, Client):
    set_random_seed()
    global client
    ci = q.get()
    client = Client(ci[0], ci[1])

def run_clients(received_info):
    try:
        #client.increase_round()
        return client.run(received_info)
    except KeyboardInterrupt:
        logging.info("exiting")
        return None

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

    cfg_path = "./cfg_files/cfg_emnist_modelsample.toml"
    #cfg_path = "./cfg_files/cfg_cifar.toml"
    
    with open(cfg_path, "r") as file:
        cfg = DotMap(toml.load(file))
    
    sim_result = {}

    sim_params = list(product(cfg.Data.alpha_list, cfg.ML.epochs_list, cfg.ML.batch_size_list))

    
    for n_clients in  cfg.Sim.n_clients:
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
            sim_result["client_number"] = n_clients
            sim_result["total_MC_it"] = cfg.Sim.total_MC_it
            sim_result["accuracy"] = np.zeros((cfg.Sim.total_MC_it, cfg.ML.communication_rounds))
            sim_result["cf_matrix"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds,cfg.Data.n_classes,cfg.Data.n_classes,))

            #model_samples = [[[] for _ in range(3)] for _ in range(cfg.Sim.total_MC_it)]
            size_cifar = 39941
            size_mnist = 487
            model_samples = np.zeros((cfg.Sim.total_MC_it, n_clients, 3, size_mnist))


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
                upper_n_clients = 500
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
                    n_clients,
                    batch_size,
                    cfg.Data.val_size,
                    upper_n_clients
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
                server = Server(server_dict, server_args)
                server_outputs = server.start() # get global model to start from

                # -----------------------------------------
                #               Setup Clients
                # -----------------------------------------
                mapping_dict = allocate_clients_to_threads(
                    cfg.ML.communication_rounds, cfg.Sim.n_threads, n_clients
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

                import os

                fd_directory = "/proc/self/fd"
                open_files_count = len(os.listdir(fd_directory))
                print(f"Number of open files: {open_files_count}")


                with Pool(
                    max_workers=np.minimum(n_clients,cfg.Sim.n_threads),
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

                        # print(np.size(client_outputs))
                        # print(np.size(client_outputs[0]["weights"]['linear.weight'].numpy()))
                        # print(np.size(client_outputs[0]["weights"]['linear.bias'].numpy()))

                        if r in {0,4,9}:
                            for ii in range(0,n_clients):
                                a = client_outputs[ii]["weights"]
                                
                                # Subsample weights from each layer
                                subsampled_weights = {}
                                for name, param in client_outputs[ii]["weights"].items():
                                    if 'weight' in name:  # only consider weight tensors, not biases
                                        # For simplicity, let's take every 10th weight (as an example)
                                        subsampled = param.data.view(-1)[::100].clone().detach()
                                        subsampled_weights[name] = subsampled
                                concatenated_tensors = torch.cat([tensor.flatten() for tensor in subsampled_weights.values()])

                                # Convert to numpy array
                                tensor_nparray = concatenated_tensors.numpy()

                                if r == 0: indx = 0
                                if r == 4: indx = 1
                                if r == 9: indx = 2
                                
                            model_samples[monte_carlo_iterr][ii][indx][:] = tensor_nparray
                            

                        # -----------------------------------------
                        #               Aggregation
                        # -----------------------------------------
                        server_outputs, acc[0, r], cf_matrix = server.run(client_outputs)
                        
                        sim_result["cf_matrix"][monte_carlo_iterr, r, :, :] = cf_matrix # store confusion matrix

                        round_end = time.time()
                        
                        logging.info(f"MC-Iteration: {monte_carlo_iterr} --- Round {r} ---  Time: {round_end - round_start} --- Accuracy: {acc[0,r]}")
                    
                    del(server_outputs)
                    sim_result["accuracy"][monte_carlo_iterr, :] = acc


            # Save the simulation results
            if "mnist" in cfg.Data.data_dir:
                prefix = "./results/MNIST_"
            elif "cifar" in cfg.Data.data_dir:
                prefix = "./results/CIFAR10_"
            suffix = f"_e-{epochs}_bs-{batch_size}_alpha-{alpha}_totalMC-{cfg.Sim.total_MC_it}.txt"
            sim_title = prefix + suffix

            sim_result['accuracy'] = sim_result['accuracy'].tolist()
            sim_result['cf_matrix'] = sim_result['cf_matrix'].tolist()
            
            with open(sim_title, "w") as convert_file:
                convert_file.write(json.dumps(sim_result))

            # Save the model samples
            if "emnist" in cfg.Data.data_dir:
                prefix = "./results/modelSamples_EMNIST_"
            elif "mnist" in cfg.Data.data_dir:
                prefix = "./results/modelSamples_MNIST_"
            elif "cifar" in cfg.Data.data_dir:
                prefix = "./results/modelSamples_CIFAR10_"
            suffix = f"n-{n_clients}_e-{epochs}_bs-{batch_size}_alpha-{alpha}_totalMC-{cfg.Sim.total_MC_it}.pkl"
            sim_title = prefix + suffix

            with open(sim_title, "wb") as f:
                pickle.dump(model_samples, f)

        # with open(sim_title, "rb") as f:
        #     model_samples = pickle.load(f)

        print(f"shape of saved model file: {np.array(model_samples).shape}")