import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = (
    False  # prevent cuda from optimizing convolutional methods
)

import toml
import numpy as np
import random
from dotmap import DotMap
import copy
from torch.multiprocessing import set_start_method, Queue
import logging
from collections import defaultdict
import time
import json
from itertools import product
# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic#:~:text=As%20of%20Python,answer%20by%20jfs
from concurrent.futures import ProcessPoolExecutor as Pool
import pickle

# methods
import data_preprocessing.data_loader as dl
import methods.fedavg as fedavg
from models.logistic_regression import logistic_regression
from data_preprocessing.data_poisoning import flip_label, random_labels, permute_labels
from models.resnet import resnet56, resnet18
from models.convnet import convnetwork
from defence.group_test import Group_Test


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

def allocate_clients_to_threads(n_rounds, n_threads, n_clients):
    mapping_dict = defaultdict(list)
    for round in range(n_rounds):

        num_clients = n_clients
        client_list = list(range(num_clients))
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

# If folder doesn't exist, create folder and store the model
def save_model(server_model_statedict, mc_iteration):
    import os
    checkpoint_folder = "./checkpoint/"
    cp_name = f"server-model_MC_{mc_iteration}_nm_{index_of_nm}"
    
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    torch.save(server_model_statedict, checkpoint_folder + cp_name + ".pt")
    
    # store the random states
    random_states = DotMap()
    random_states.random_state = random.getstate()
    random_states.np_random_state = np.random.get_state()
    random_states.torch_random_state = torch.random.get_rng_state()
    random_states.torch_cuda_random_state_all = torch.cuda.random.get_rng_state_all()
    with open(checkpoint_folder + cp_name + "_random_states.pickle", 'wb') as handle:
        pickle.dump(random_states, handle, protocol=pickle.HIGHEST_PROTOCOL)

    checkpoint_exists = True
    return checkpoint_exists

# load the server model right before GT
def load_model(mc_iteration, index_of_nm):
    checkpoint_folder = "./checkpoint/"
    cp_name = f"server-model_MC_{mc_iteration}"
    server_model_statedict = torch.load(checkpoint_folder + f"/server-model_MC_{mc_iteration}_nm_{index_of_nm}.pt")
    
    # load pickled file and set random states as before checkpoint
    with open(checkpoint_folder + cp_name + "_random_states.pickle", 'rb') as handle:
        random_states = pickle.load(handle)
        random.setstate(random_states.random_state)
        np.random.set_state(random_states.np_random_state)
        torch.set_rng_state(random_states.torch_random_state)
        # loop through the devices and set the random states for each device 
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_rng_state(random_states.torch_cuda_random_state_all[i], device=i)
        
    return server_model_statedict

if __name__ == "__main__":

    cfg_path = "./cfg_files/cfg_cifar.toml"
    with open(cfg_path, "r") as file:
        cfg = DotMap(toml.load(file))
    
    # keep track if checkpoint is stored for given MC iteration
    checkpoint_exists = [[False for x in range(cfg.Sim.total_MC_it)] for y in range(len(cfg.Sim.n_malicious_list))]
    
    sim_result = {}

    sim_params = list(product(
            cfg.Data.alpha_list,
            cfg.ML.epochs_list,
            cfg.Sim.n_malicious_list,
            cfg.ML.batch_size_list,
            cfg.GT.crossover_probability_list,
            cfg.Sim.MODE_list,
            cfg.Sim.attack_list)
            )

    for (
        alpha,
        epochs,
        n_malicious,
        batch_size,
        crossover_probability,
        MODE,
        ATTACK,
    ) in sim_params:
        
        threshold_vec = np.arange(
            cfg.GT.BCJR_min_threshold,
            cfg.GT.BCJR_max_threshold,
            cfg.GT.BCJR_step_threshold,
        ).tolist()
        
        noiseless_gt = True if MODE == 3 else False
        no_defence = True if MODE == 0 else False
        oracle = True if MODE == 1 else False

        
        # No need to loop over thresholds if we dont do group testing
        if oracle or no_defence:
            threshold_vec = [np.inf]

        # prepare to store results
        sim_result["epochs"] = epochs
        sim_result["val_size"] = cfg.Data.val_size
        sim_result["test_threshold"] = cfg.GT.test_threshold
        sim_result["group_test_round"] = cfg.GT.group_test_round
        sim_result["batch_size"] = batch_size
        sim_result["alpha"] = alpha
        sim_result["n_malicious"] = n_malicious
        sim_result["data_dir"] = cfg.Data.data_dir
        sim_result["lr"] = cfg.ML.lr
        sim_result["wd"] = cfg.ML.wd
        sim_result["momentum"] = cfg.ML.momentum
        sim_result["comm_round"] = cfg.ML.communication_rounds
        sim_result["client_number"] = cfg.Sim.n_clients
        sim_result["total_MC_it"] = cfg.Sim.total_MC_it
        sim_result["threshold_vec"] = threshold_vec
        sim_result["group_acc"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, cfg.GT.n_tests))
        sim_result["group_prec"] = np.zeros((len(threshold_vec),cfg.Sim.total_MC_it,cfg.GT.n_tests,cfg.Data.n_classes,))
        sim_result["group_recall"] = np.zeros((len(threshold_vec),cfg.Sim.total_MC_it,cfg.GT.n_tests,cfg.Data.n_classes,))
        sim_result["group_f1"] = np.zeros((len(threshold_vec),cfg.Sim.total_MC_it,cfg.GT.n_tests,cfg.Data.n_classes,))
        sim_result["bsc_channel"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, 2, 2))
        sim_result["malicious_clients"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, cfg.Sim.n_clients))
        sim_result["DEC"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, cfg.Sim.n_clients))
        sim_result["syndrome"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, cfg.GT.n_tests))
        sim_result["accuracy"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, cfg.ML.communication_rounds))
        sim_result["cf_matrix"] = np.zeros((len(threshold_vec),cfg.Sim.total_MC_it,cfg.ML.communication_rounds,cfg.Data.n_classes,cfg.Data.n_classes,))

        try:
            set_start_method("spawn")
        except RuntimeError:
            pass

        accuracy = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, cfg.ML.communication_rounds))
        P_FA = np.zeros(len(threshold_vec))
        P_MD = np.zeros(len(threshold_vec))

        for thres_indx, threshold_dec in enumerate(threshold_vec):
            logging.info("Starting with threshold_dec : {}".format(threshold_dec))
            FA = 0
            MD = 0
            

            for monte_carlo_iterr in range(cfg.Sim.total_MC_it):
                set_random_seed(monte_carlo_iterr)  # all mc iterations should have same seed for each threshold value
                # -----------------------------------------
                #           Create attacks
                # -----------------------------------------
                malicious_clients = np.random.permutation(cfg.Sim.n_clients)
                malicious_clients = malicious_clients[:n_malicious].tolist()
                defective = np.zeros((1, cfg.Sim.n_clients), dtype=np.uint8)
                defective[:, malicious_clients] = 1
                attacks = list_of_lists = [[] for i in range(cfg.Sim.n_clients)]
                for client in range(cfg.Sim.n_clients):
                    if client in malicious_clients:
                        if ATTACK == 0:
                            attacks[client].append((permute_labels,))
                        elif ATTACK == 1:
                            attacks[client].append((random_labels,))
                        elif ATTACK == 2:
                            if "mnist" in cfg.Data.data_dir:
                                src = 1
                                target = 7
                            elif "cifar10" in cfg.Data.data_dir:
                                src = 7
                                target = 4
                            label_flips = [(src, target)]
                            attacks[client].append((flip_label, label_flips))

                sim_result["malicious_clients"][thres_indx, monte_carlo_iterr, :] = np.array(defective)
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
                    attacks,
                    cfg.Data.val_size,
                )

                # -----------------------------------------
                #         Choose Model and FL protocol
                # -----------------------------------------
                if "cifar" in cfg.Data.data_dir:
                    # Model = resnet18
                    Model = convnetwork
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
                
                # get global model to start from
                index_of_nm = cfg.Sim.n_malicious_list.index(n_malicious)
                if not checkpoint_exists[index_of_nm][monte_carlo_iterr]:
                    server_outputs = server.start()
                    start_round = 0
                elif checkpoint_exists[index_of_nm][monte_carlo_iterr] and (oracle is False):
                    checkpoint_model_statedict = load_model(monte_carlo_iterr, index_of_nm)
                    server_outputs = [checkpoint_model_statedict for x in range(server.n_threads)]
                    start_round = cfg.GT.group_test_round
                    

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
                #          Make Group Test Object
                # -----------------------------------------
                if not (oracle or no_defence):
                    
                    P_FA_test = 1e-6 if noiseless_gt else cfg.GT.P_FA
                    P_MD_test = 1e-6 if noiseless_gt else cfg.GT.P_MD
                    prevalence = n_malicious / cfg.Sim.n_clients 
                    
                    gt = Group_Test(
                        cfg.Sim.n_clients,
                        prevalence,
                        threshold_dec,
                        min_acc=0,
                        threshold_from_max_acc=cfg.GT.test_threshold,
                        P_FA_test=P_FA_test,
                        P_MD_test=P_MD_test,
                    )
                    sim_result["bsc_channel"][thres_indx, monte_carlo_iterr, :, :] = gt.ChannelMatrix
                    syndrome = np.matmul(defective, gt.parity_check_matrix.transpose())

                # -----------------------------------------
                #            Main Loop
                # -----------------------------------------
                # each thread will create a client object containing the client information
                acc = np.zeros((1, cfg.ML.communication_rounds))
                all_class_malicious = False
                with Pool(
                    max_workers=cfg.Sim.n_threads,
                    initializer=init_process,
                    initargs=(client_info, Client),
                ) as pool:
                    for r in range(start_round, cfg.ML.communication_rounds):
                        round_start = time.time()
                        
                        # Store the server model before GT to be used in the other loops
                        if (not checkpoint_exists[index_of_nm][monte_carlo_iterr]) and (r == cfg.GT.group_test_round):
                            checkpoint_exists[index_of_nm][monte_carlo_iterr] = save_model(server_outputs[0], monte_carlo_iterr, index_of_nm)

                        # -----------------------------------------
                        #         Perform local training
                        # -----------------------------------------
                        client_outputs = pool.map(run_clients, server_outputs)
                        client_outputs = [c for sublist in client_outputs for c in sublist]
                        client_outputs.sort(key=lambda tup: tup["client_index"])

                        if no_defence:
                            # if no defence, keep all clients
                            clients_to_aggregate = client_outputs
                        
                        elif oracle:
                            # if oracle, only keep benign clients
                            clients_to_aggregate = []
                            for i in range(cfg.Sim.n_clients):
                                if i not in malicious_clients:
                                    clients_to_aggregate.append(client_outputs[i])
                        else:
                            # -----------------------------------------
                            #           Group Testing
                            # -----------------------------------------
                            if r < cfg.GT.group_test_round:
                                DEC = np.zeros((1, cfg.Sim.n_clients), dtype=np.uint8)
                            elif r == cfg.GT.group_test_round:  
                                (
                                    group_accuracies,
                                    prec,
                                    rec,
                                    f1,
                                ) = gt.get_group_accuracies(client_outputs, server)

                                if noiseless_gt is True:
                                    DEC = gt.noiseless_group_test(syndrome)
                                else:
                                    if ATTACK < 2:
                                        DEC = gt.perform_group_test(group_accuracies)
                                    elif ATTACK == 2:
                                        DEC = gt.perform_group_test(rec[:, src])

                                    DEC = gt.perform_group_test(group_accuracies)
                                MD = MD + np.sum(gt.DEC[defective == 1] == 0)
                                FA = FA + np.sum(gt.DEC[defective == 0] == 1)
                                if (
                                    np.sum(DEC) == DEC.shape[1]
                                ):  # , "All are classified as malicious"
                                    all_class_malicious = True
                                    
                                sim_result["group_acc"][thres_indx, monte_carlo_iterr, :] = group_accuracies
                                sim_result["group_prec"][thres_indx, monte_carlo_iterr, :, :] = prec
                                sim_result["group_recall"][thres_indx, monte_carlo_iterr, :, :] = rec
                                sim_result["group_f1"][thres_indx, monte_carlo_iterr, :, :] = f1
                                sim_result["DEC"][thres_indx, monte_carlo_iterr, :] = DEC
                                sim_result["syndrome"][thres_indx, monte_carlo_iterr] = syndrome[0]
                            # -----------------------------------------
                            #               Aggregation
                            # -----------------------------------------
                            # If all malicious, just use all
                            if all_class_malicious == True:
                                clients_to_aggregate = client_outputs
                            else:
                                clients_to_aggregate = [
                                    client_outputs[client_idx]
                                    for client_idx in range(cfg.Sim.n_clients)
                                    if DEC[:, client_idx] == 0
                                ]
                            # -----------------------------------------
                            #               Aggregation
                            # -----------------------------------------
                            # If all malicious, just use all
                            if all_class_malicious == True:
                                clients_to_aggregate = client_outputs
                            else:
                                clients_to_aggregate = [
                                    client_outputs[client_idx]
                                    for client_idx in range(cfg.Sim.n_clients)
                                    if DEC[:, client_idx] == 0
                                ]
                        server_outputs, acc[0, r], cf_matrix = server.run(
                            clients_to_aggregate
                        )

                        sim_result["cf_matrix"][
                            thres_indx, monte_carlo_iterr, r, :, :
                        ] = cf_matrix

                        round_end = time.time()
                        logging.info(f"Threshold: {threshold_dec} --- MC-Iteration: {monte_carlo_iterr} --- Round {r} ---  Time: {round_end - round_start} --- Accuracy: {acc[0,r]}")

                    accuracy[thres_indx, monte_carlo_iterr, :] = acc

            if n_malicious > 0:
                P_MD[thres_indx] = MD / (n_malicious * cfg.Sim.total_MC_it)
            P_FA[thres_indx] = FA / (
                (cfg.Sim.n_clients - n_malicious) * cfg.Sim.total_MC_it
            )

            # make all nparrays JSON serializable
            checkpoint_dict = copy.deepcopy(sim_result)

            checkpoint_dict["accuracy"] = accuracy.tolist()
            checkpoint_dict["P_MD"] = P_MD.tolist()
            checkpoint_dict["P_FA"] = P_FA.tolist()
            checkpoint_dict["group_acc"] = checkpoint_dict["group_acc"].tolist()
            checkpoint_dict["group_prec"] = checkpoint_dict["group_prec"].tolist()
            checkpoint_dict["group_recall"] = checkpoint_dict["group_recall"].tolist()
            checkpoint_dict["group_f1"] = checkpoint_dict["group_f1"].tolist()
            checkpoint_dict["cf_matrix"] = checkpoint_dict["cf_matrix"].tolist()
            checkpoint_dict["DEC"] = checkpoint_dict["DEC"].tolist()
            checkpoint_dict["syndrome"] = checkpoint_dict["syndrome"].tolist()
            checkpoint_dict["malicious_clients"] = checkpoint_dict["malicious_clients"].tolist()
            checkpoint_dict["bsc_channel"] = checkpoint_dict["bsc_channel"].tolist()

            if "mnist" in cfg.Data.data_dir:
                prefix = "./results/MNIST_"
            elif "cifar" in cfg.Data.data_dir:
                prefix = "./results/CIFAR10_"
            suffix = f"m-{n_malicious},{cfg.Sim.n_clients}_e-{epochs}_bs-{batch_size}_alpha-{alpha}_totalMC-{cfg.Sim.total_MC_it}_MODE-{MODE}_att-{ATTACK}.txt"
            sim_title = prefix + suffix

            with open(sim_title, "w") as convert_file:
                convert_file.write(json.dumps(checkpoint_dict))
