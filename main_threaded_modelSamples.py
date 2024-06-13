import toml
import random
import logging
from collections import defaultdict
import math
import time
import json
import pickle
from itertools import product
import copy
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = (
    False  # prevent cuda from optimizing convolutional methods
)
from dotmap import DotMap

import numpy as np

# methods
import data_preprocessing.data_loader as dl
import methods.fedavg as fedavg
from models.logistic_regression import logistic_regression
from models.eff_net import efficient_net
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method

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

def local_training(clients):
    result = []
    for client in clients:
        result.append(client.run())
    return result 

def update_global_model(clients, server_model):
    for client in clients:
        client.load_model(server_model)

def subsample_weights(model, subsample_rate = 1):
    subsampled_weights = {}
    for name, param in model.items():
        if ("weight" in name or "bias" in name):  # consider weight and bias tensors
            subsample_rate = 1 # subsampling rate
            subsampled = (param.data.view(-1)[::subsample_rate].clone().detach())
            subsampled_weights[name] = subsampled
    concatenated_tensors = torch.cat([tensor.flatten()for tensor in subsampled_weights.values()])
    # Convert to numpy array
    model_weights = concatenated_tensors.numpy()
    return model_weights

if __name__ == "__main__":
    
    import pickle


    # with open('./results/modelSamples_ADULT_n-100_e-1_bs-64_alpha-inf_totalMC-5000_43.pkl', 'rb') as f:
    #     d = pickle.load(f)
    
    cfg_path = "./cfg_files/cfg_emnist_modelsample.toml"
   # cfg_path = "./cfg_files/cfg_mnist_modelsample.toml"
   # cfg_path = "./cfg_files/cfg_cifar.toml"

    with open(cfg_path, "r") as file: cfg = DotMap(toml.load(file))

    sim_result = {}
    sim_params = list(product(cfg.Data.alpha_list, cfg.ML.epochs_list, cfg.ML.batch_size_list))
    
    if "emnist" in cfg.Data.data_dir:
        prefix = "./results/modelSamples_EMNIST_"
    elif "mnist" in cfg.Data.data_dir:
        prefix = "./results/modelSamples_MNIST_"
    elif "cifar" in cfg.Data.data_dir:
        prefix = "./results/modelSamples_CIFAR10_"
    elif "adult" in cfg.Data.data_dir:
        prefix = "./results/modelSamples_ADULT_"

    save_n = 0
    for n_clients in cfg.Sim.n_clients:
        for alpha, epochs, batch_size in sim_params:
            if cfg.Data.partition_method == "homo":
                alpha = np.inf
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
            sim_result["accuracy"] = np.zeros((cfg.Sim.n_init_points, cfg.Sim.total_MC_it, cfg.ML.communication_rounds))
            sim_result["cf_matrix"] = np.zeros((cfg.Sim.n_init_points, cfg.Sim.total_MC_it,cfg.ML.communication_rounds,cfg.Data.n_classes,cfg.Data.n_classes,))

            # model_samples = [[[] for _ in range(3)] for _ in range(cfg.Sim.total_MC_it)]
            size_cifar = 39941
            size_mnist = 7850
            size_emnist = 7850
            size_adult = 210
            
            if "cifar" in cfg.Data.data_dir:
                size_model = size_cifar
            elif "emnist" in cfg.Data.data_dir:
                size_model = size_emnist
            elif "mnist" in cfg.Data.data_dir:
                size_model = size_mnist
            elif "adult" in cfg.Data.data_dir:
                size_model = size_adult
                
            n_per_save = 250
            save_indx = 0
            rounds_to_save = [0]
            n_rounds_to_save = len(rounds_to_save)
            model_samples = np.zeros((n_per_save, n_clients, n_rounds_to_save, size_model))
            init_model_samples = np.zeros((cfg.Sim.n_init_points, size_model))

            try:
                set_start_method("spawn")
            except RuntimeError:
                pass

            accuracy = np.zeros((cfg.Sim.total_MC_it, cfg.ML.communication_rounds))

            logging.info("Starting Simulation")
            for init_pt_iter in range(cfg.Sim.n_init_points):
                
                set_random_seed(init_pt_iter)
                if "cifar" in cfg.Data.data_dir:
                    Model = efficient_net
                elif "emnist" in cfg.Data.data_dir:
                    Model = logistic_regression
                elif "mnist" in cfg.Data.data_dir:
                    Model = logistic_regression
                elif "adult" in cfg.Data.data_dir:
                    Model = logistic_regression

                Server = fedavg.Server
                server_dict = DotMap()
                start_model = Model(cfg.Data.n_classes, datadir=cfg.Data.data_dir)
                init_model_samples[init_pt_iter][:] = subsample_weights(start_model.cpu().state_dict())
                
                suffix_init_models = f"start_models_n-{n_clients}_e-{epochs}_bs-{batch_size}_alpha-{alpha}_totalMC-{cfg.Sim.total_MC_it}.pkl"
                sim_title = prefix + suffix_init_models
                with open(sim_title, "wb") as f:
                    pickle.dump(init_model_samples, f)
                
                for monte_carlo_iterr in range(cfg.Sim.total_MC_it):
                    logging.info("Monte Carlo Iteration: {}".format(monte_carlo_iterr))
                    set_random_seed(monte_carlo_iterr)
                    # -----------------------------------------
                    # Obtain dataset for server and the clients
                    # -----------------------------------------
                    upper_n_clients = n_clients
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
                        upper_n_clients,
                    )

                    # -----------------------------------------
                    #         Choose Model and FL protocol
                    # -----------------------------------------
                    # Pick FL method
                    Client = fedavg.Client

                    # -----------------------------------------
                    #               Setup Server
                    # -----------------------------------------
                    server_dict.model = copy.deepcopy(start_model)
                    server_dict.val_data = server_val_dl
                    server_dict.test_data = server_test_dl
                    server_dict.num_classes = cfg.Data.n_classes

                    server_args = DotMap()
                    server_args.n_threads = cfg.Sim.n_threads
                    server = Server(server_dict, server_args)
                    server_outputs = server.start()  # get global model to start from

                    # -----------------------------------------
                    #               Setup Clients
                    # -----------------------------------------
                    client_dict = [
                        {
                            "idx": i,
                            "data_dir": cfg.Data.data_dir,
                            "train_data": train_data_local_dict[i],
                            "device": "cuda:{}".format(i % torch.cuda.device_count())
                            if torch.cuda.is_available() else "cpu",
                            "model_type": Model,
                            "num_classes": cfg.Data.n_classes,
                        }
                        for i in range(n_clients)
                    ]
                    # init nodes
                    client_args = DotMap()
                    client_args.epochs = epochs
                    client_args.batch_size = batch_size
                    client_args.lr = cfg.ML.lr
                    client_args.momentum = cfg.ML.momentum
                    client_args.wd = cfg.ML.wd

                    clients = [Client(client_dict[i], client_args) for i in range(n_clients)]
                    
                    # -----------------------------------------
                    #            Main Loop
                    # -----------------------------------------
                    # each thread will create a client object containing the client information
                    acc = np.zeros((1, cfg.ML.communication_rounds))

                    # Print number of open files
                    import os
                    fd_directory = "/proc/self/fd"
                    open_files_count = len(os.listdir(fd_directory))
                    print(f"Number of open files: {open_files_count}")

                    with mp.Pool(cfg.Sim.n_threads) as p:
                        client_splits = np.array_split(clients, cfg.Sim.n_threads)
                        for r in range(0, cfg.ML.communication_rounds):
                            round_start = time.time()
                            # -----------------------------------------
                            #         Perform local training
                            # -----------------------------------------
                            update_global_model(clients, server_outputs) # set the new server model in the clients
                            client_outputs = p.map(local_training, client_splits)
                            client_outputs = [c for sublist in client_outputs for c in sublist]
                            client_outputs.sort(key=lambda tup: tup["client_index"])
                            
                            if r in rounds_to_save:
                                indx = rounds_to_save.index(r)
                                for ii in range(0, n_clients):
                                    client_weights = subsample_weights(client_outputs[ii]["weights"])
                                    model_samples[save_indx][ii][indx][:] = client_weights

                            # -----------------------------------------
                            #               Aggregation
                            # -----------------------------------------
                            server_outputs, acc[0, r], cf_matrix = server.run(client_outputs)
                            sim_result["cf_matrix"][init_pt_iter, monte_carlo_iterr, r, :, :] = cf_matrix  # store confusion matrix
                            round_end = time.time()

                            logging.info(f"(init_pt, MC_iter)): ({init_pt_iter},{monte_carlo_iterr}) --- Round {r} ---  Time: {round_end - round_start} --- Accuracy: {acc[0,r]}")

                        # Shutdown the pool
                        del server_outputs

                        if (monte_carlo_iterr + 1) % n_per_save == 0:
                            suffix = f"n-{n_clients}_e-{epochs}_bs-{batch_size}_alpha-{alpha}_totalMC-{cfg.Sim.total_MC_it}_{save_n}.pkl"
                            sim_title = prefix + suffix
                            
                            save_n += 1

                            with open(sim_title, "wb") as f:
                                pickle.dump(model_samples, f)
                            save_indx = 0
                            print(
                                f"shape of saved model file: {np.array(model_samples).shape}"
                            )
                        else:
                            save_indx += 1

                        sim_result["accuracy"][init_pt_iter, monte_carlo_iterr, :] = acc

        # Save the simulation results
        if "emnist" in cfg.Data.data_dir:
            prefix = "./results/EMNIST_"
        elif "mnist" in cfg.Data.data_dir:
            prefix = "./results/MNIST_"
        elif "cifar" in cfg.Data.data_dir:
            prefix = "./results/CIFAR10_"
        elif "adult" in cfg.Data.data_dir:
            prefix = "./results/ADULT_"
            
        suffix = f"results_n-{n_clients}_e-{epochs}_bs-{batch_size}_alpha-{alpha}_totalMC-{cfg.Sim.total_MC_it}.txt"
        sim_title = prefix + suffix

        sim_result["accuracy"] = sim_result["accuracy"].tolist()
        sim_result["cf_matrix"] = sim_result["cf_matrix"].tolist()

        with open(sim_title, "w") as convert_file:
            convert_file.write(json.dumps(sim_result))

            # with open(sim_title, "rb") as f:
            #     model_samples = pickle.load(f)
