import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = (
    False  # prevent cuda from optimizing convolutional methods
)

import numpy as np
import random
import copy
from torch.multiprocessing import set_start_method, Queue
import logging
logging.getLogger('timm').setLevel(logging.WARNING) #ERROR
from collections import defaultdict
import time
import json
import subprocess
from itertools import product
# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic#:~:text=As%20of%20Python,answer%20by%20jfs
from concurrent.futures import ProcessPoolExecutor as Pool
import pickle

# methods
import data_preprocessing.data_loader as dl
import methods.fedavg as fedavg
from models.logistic_regression import logistic_regression
from data_preprocessing.data_poisoning import flip_label, random_labels, permute_labels
from models.resnet import resnet18
from models.eff_net import efficient_net
from defence.group_test import Group_Test
import toml
from dotmap import DotMap

import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method

check_for_stupid_error = 0

# Helper Functions
def local_training(clients):
    result = []
    for client in clients:
        result.append(client.run())
    return result 

def update_global_model(clients, server_model):
    for client in clients:
        client.load_model(server_model)


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
def save_model(server_model_statedict, mc_iteration, index_of_nm):
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
    cp_name = f"server-model_MC_{mc_iteration}_nm_{index_of_nm}"
    server_model_statedict = torch.load(checkpoint_folder + cp_name + ".pt")
    
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
    prev_and_cross_sim = cfg.GT.manual_params
    logging.info(f"The value of prevalence simulation is {prev_and_cross_sim}")
    
    # if no key is given, make sure we are using all clients
    n_clients_total = cfg.Sim.n_clients_total if "n_clients_total" in cfg.Sim else cfg.Sim.n_clients
        
    if not prev_and_cross_sim:
        assert len(cfg.GT.crossover_probability_list) == 1, "crossover_probability_list should be of length 1"
        assert len(cfg.GT.prevalence_list) == 1, "prevalence_list should be of length 1"
        
    kick_out_clients = cfg.Oracle.kick_out_clients
    logging.info(f"The value of kicking out clients is {kick_out_clients}")
    if kick_out_clients:
        assert len(cfg.Sim.MODE_list) == 1, "For kicking out clients you only need one MODE 1 (Oracle)"
        assert cfg.Sim.MODE_list[0] == 1, "For kicking out clients you only need MODE 1 (Oracle)"
        #assert len(cfg.Sim.n_malicious_list) == 1, "For kicking out clients you need only 0 malicious_clients"
        #assert cfg.Sim.n_malicious_list[0] == 0, "For kicking out clients you need only 0 malicious_clients"
        assert "isic" in cfg.Data.data_dir, "For kicking out clients only isic dataset is allowed for now"

    sim_params = list(product(
            cfg.Data.alpha_list,
            cfg.ML.epochs_list,
            cfg.Sim.n_malicious_list,
            cfg.ML.batch_size_list,
            cfg.GT.crossover_probability_list,
            cfg.GT.prevalence_list,
            cfg.Sim.MODE_list,
            cfg.Sim.attack_list)
            )

    for (
        alpha,
        epochs,
        n_malicious,
        batch_size,
        crossover_probability,
        prevalence_sim,
        MODE,
        ATTACK,
    ) in sim_params:
        
        if ATTACK == 2 and n_malicious == 0 and MODE == 2:
            if "mnist" in cfg.Data.data_dir:
                src = 1
                target = 7
            elif "cifar10" in cfg.Data.data_dir:
                src = 7
                target = 4
            elif "isic" in cfg.Data.data_dir:
                src = 0
                target = 1
        
        ## Lambda
        threshold_vec = np.arange(
            cfg.GT.BCJR_min_threshold,
            cfg.GT.BCJR_max_threshold,
            cfg.GT.BCJR_step_threshold,
        ).tolist()        

        no_PCA_components = cfg.PCA.no_comp
        if cfg.Sim.PCA_simulation is True:
            print(f"Number of PCA components = {no_PCA_components}")
            sim_result["no_pca_comp"] = no_PCA_components
            assert MODE == 2, "Only Mode 2 is supported for PCA sims!"
            assert len(threshold_vec) == 1, "Only use one threshold please for PCA sims!"
            assert cfg.GT.group_test_round > cfg.ML.communication_rounds, "GT round should be done later than Comm rounds for PCA sims!"

        noiseless_gt = True if MODE == 3 else False
        no_defence = True if MODE == 0 else False
        oracle = True if MODE == 1 else False
        GM_aggregation = True if MODE == 4 else False

        
        # No need to loop over thresholds if we dont do group testing
        if oracle or no_defence or GM_aggregation:
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
        sim_result["client_number"] = n_clients_total
        sim_result["total_MC_it"] = cfg.Sim.total_MC_it
        sim_result["threshold_vec"] = threshold_vec
        sim_result["group_acc"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, cfg.GT.n_tests))
        sim_result["group_prec"] = np.zeros((len(threshold_vec),cfg.Sim.total_MC_it,cfg.GT.n_tests,cfg.Data.n_classes,))
        sim_result["group_recall"] = np.zeros((len(threshold_vec),cfg.Sim.total_MC_it,cfg.GT.n_tests,cfg.Data.n_classes,))
        sim_result["group_f1"] = np.zeros((len(threshold_vec),cfg.Sim.total_MC_it,cfg.GT.n_tests,cfg.Data.n_classes,))
        sim_result["bsc_channel"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, 2, 2))
        sim_result["malicious_clients"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, n_clients_total))
        sim_result["DEC"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, cfg.Sim.n_clients))
        sim_result["LLRO"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, cfg.Sim.n_clients))
        sim_result["LLRin"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, cfg.Sim.n_clients))
        sim_result["td_used"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it))
        sim_result["nm_est"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it))
        sim_result["syndrome"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, cfg.GT.n_tests))
        sim_result["test_values"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, cfg.GT.n_tests))
        sim_result["accuracy"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, cfg.ML.communication_rounds))
        sim_result["cf_matrix"] = np.zeros((len(threshold_vec),cfg.Sim.total_MC_it,cfg.ML.communication_rounds,cfg.Data.n_classes,cfg.Data.n_classes,))
        sim_result["ss_thres"] = cfg.Test.ss_thres
        no_clusters = 5 ## HARD_CODED
        sim_result["s_scores"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, no_clusters))
        sim_result["d_scores"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, no_clusters))
        print(f"Silh_score thresh - {cfg.Test.ss_thres}")
        print(f"DELTA - {cfg.GT.DELTA}")
        if kick_out_clients is True:
            sim_result["oracle_kicked_clients"] = [] 
        if cfg.Sim.PCA_simulation is True:
            sim_result["cosine_similarity_per_label"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds,cfg.GT.n_tests,cfg.Data.n_classes))
            sim_result["cosine_similarity_model"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds,cfg.GT.n_tests))
            sim_result["PCA_components"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds,cfg.GT.n_tests,no_PCA_components))
            sim_result["PCA_variance"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds,no_PCA_components))
            sim_result["PCA_variance_ratio"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds,no_PCA_components))
            sim_result["PCA_components_per_label"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds,cfg.Data.n_classes ,cfg.GT.n_tests, no_PCA_components))
            sim_result["PCA_variance_per_label"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds,cfg.Data.n_classes , no_PCA_components))  
            sim_result["PCA_variance_ratio_per_label"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds,cfg.Data.n_classes , no_PCA_components))     
            sim_result["group_acc"] = np.zeros((len(threshold_vec), cfg.Sim.total_MC_it, cfg.ML.communication_rounds, cfg.GT.n_tests))
            sim_result["group_prec"] = np.zeros((len(threshold_vec),cfg.Sim.total_MC_it, cfg.ML.communication_rounds, cfg.GT.n_tests, cfg.Data.n_classes,))
            sim_result["group_recall"] = np.zeros((len(threshold_vec),cfg.Sim.total_MC_it, cfg.ML.communication_rounds, cfg.GT.n_tests, cfg.Data.n_classes,))
            sim_result["group_f1"] = np.zeros((len(threshold_vec),cfg.Sim.total_MC_it, cfg.ML.communication_rounds, cfg.GT.n_tests, cfg.Data.n_classes,))
            sim_result["loss_per_label"] = np.zeros((len(threshold_vec),cfg.Sim.total_MC_it, cfg.ML.communication_rounds, cfg.GT.n_tests, cfg.Data.n_classes,))
        else:
            sim_result["PCA_components"] = np.zeros((cfg.Sim.total_MC_it,cfg.GT.n_tests,no_PCA_components))
            sim_result["PCA_components_per_label"] = np.zeros((cfg.Sim.total_MC_it, cfg.Data.n_classes, cfg.GT.n_tests, no_PCA_components))

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
                malicious_clients = np.random.permutation(n_clients_total)
                malicious_clients = malicious_clients[:n_malicious].tolist()
                print(f"malicious clients: {malicious_clients} \n")
                if kick_out_clients is True:
                    temp_list = list( set(malicious_clients) & set(cfg.Oracle.which_ones) )
                    assert len(temp_list) == 0, "The which ones list should not intersect malicious clients"
                defective = np.zeros((1, n_clients_total), dtype=np.uint8)
                defective[:, malicious_clients] = 1 # 
                attacks = list_of_lists = [[] for i in range(n_clients_total)]
                for client in range(n_clients_total):
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
                            elif "isic" in cfg.Data.data_dir:
                                src = 0 # melanoma
                                target = 1 # mole
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
                    n_clients_total,
                    batch_size,
                    attacks,
                    cfg.Data.val_size,
                )

                # -----------------------------------------
                #         Choose Model and FL protocol
                # -----------------------------------------
                if "cifar" in cfg.Data.data_dir:
                    Model = resnet18
                elif "mnist" in cfg.Data.data_dir:
                    Model = logistic_regression
                elif "isic" in cfg.Data.data_dir:
                    Model = efficient_net

                # Pick FL method
                Server = fedavg.Server
                Client = fedavg.Client

                # -----------------------------------------
                #               Setup Server
                # -----------------------------------------
                server_dict = DotMap()
                server_dict.model_type = Model
                server_dict.val_data = server_val_dl
                server_dict.test_data = server_test_dl
                server_dict.num_classes = cfg.Data.n_classes

                server_args = DotMap()
                server_args.n_threads = cfg.Sim.n_threads
                server_args.aggregation = "GM" if GM_aggregation is True else "Avg"
                server = Server(server_dict, server_args)
                    
                print(f"Alpha is equal to {alpha} and aggregation method is {server_args.aggregation}")
                
                
                # get global model to start from
                index_of_nm = cfg.Sim.n_malicious_list.index(n_malicious)
                if not checkpoint_exists[index_of_nm][monte_carlo_iterr]:
                    server_outputs = server.start()
                    start_round = 0
                elif checkpoint_exists[index_of_nm][monte_carlo_iterr] and (not oracle):
                    checkpoint_model_statedict = load_model(monte_carlo_iterr, index_of_nm)
                    server_outputs = [checkpoint_model_statedict for x in range(server.n_threads)]
                    start_round = cfg.GT.group_test_round
                    

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
                    for i in range(n_clients_total)
                ]
                # init nodes
                client_args = DotMap()
                client_args.epochs = epochs
                client_args.batch_size = batch_size
                client_args.lr = cfg.ML.lr
                client_args.momentum = cfg.ML.momentum
                client_args.wd = cfg.ML.wd
                
                clients = [Client(client_dict[i], client_args) for i in range(n_clients_total)]

                # -----------------------------------------
                #          Make Group Test Object
                # -----------------------------------------
                if not (oracle or no_defence or GM_aggregation):
                    if not prev_and_cross_sim:
                        P_FA_test = 1e-6 if noiseless_gt else cfg.GT.P_FA
                        P_MD_test = 1e-6 if noiseless_gt else cfg.GT.P_MD
                        prevalence = n_malicious / cfg.Sim.n_clients 
                    else:
                        P_FA_test = crossover_probability
                        P_MD_test = crossover_probability
                        prevalence = prevalence_sim
                        logging.info(f"I am choosing Mismatched for prev{prevalence} and P_FA_test {P_FA_test} and P_MD_test {P_MD_test}!")
                    gt = Group_Test(
                        cfg.Sim.n_clients,
                        cfg.GT.n_tests,
                        prevalence,
                        threshold_dec,
                        min_acc=0,
                        threshold_from_max_acc=cfg.GT.test_threshold,
                        P_FA_test=P_FA_test,
                        P_MD_test=P_MD_test,
                    )
                    sim_result["bsc_channel"][thres_indx, monte_carlo_iterr, :, :] = gt.ChannelMatrix

                # -----------------------------------------
                #            Main Loop
                # -----------------------------------------
                # each thread will create a client object containing the client information
                acc = np.zeros((1, cfg.ML.communication_rounds))
                all_class_malicious = False
                malicious_counter = np.zeros((1, n_clients_total))
                with mp.Pool(cfg.Sim.n_threads) as p:
                    for r in range(start_round, cfg.ML.communication_rounds):
                        round_start = time.time()
                        
                        # set the new server model in the clients
                        update_global_model(clients, server_outputs) 
                        
                        # -----------------------------------------
                        #         Perform local training
                        # -----------------------------------------
                        # randomly sample n_clients from the total clients without replacement
                        sampled_clients_indices = np.random.choice(n_clients_total, cfg.Sim.n_clients, replace=False)
                        sampled_clients = [clients[i] for i in sampled_clients_indices]
                        # assign clients to threads
                        client_splits = np.array_split(sampled_clients, cfg.Sim.n_threads) 
                        # perform local training
                        client_outputs = p.map(local_training, client_splits) 
                        # flatten the list of lists
                        client_outputs = [c for sublist in client_outputs for c in sublist]
                        # sort the list of dictionaries by the client index
                        client_outputs.sort(key=lambda tup: tup["client_index"])

                        if len(client_outputs) != cfg.Sim.n_clients or len(set([client_outputs[i]["client_index"] for i in range(len(client_outputs))])) != cfg.Sim.n_clients:
                            print("No PASS!")
                            print(f"Problem with threads - it happend {check_for_stupid_error + 1} times")
                            check_for_stupid_error = check_for_stupid_error + 1 
                            assert False, "Vari karin o miku!!!"

                        if no_defence:
                            # if no defence, keep all clients
                            clients_to_aggregate = client_outputs
                        
                        elif GM_aggregation:
                            # if doing GM aggregation, keep all of them
                            clients_to_aggregate = client_outputs
        
                        elif oracle:
                            # if oracle, only keep benign clients
                            clients_to_aggregate = []
                            for i in range(cfg.Sim.n_clients):
                                if i not in malicious_clients:
                                    clients_to_aggregate.append(client_outputs[i])
                            if kick_out_clients:
                                clients_to_aggregate = []
                                clients_to_kick_out = []
                                for i in range(cfg.Sim.n_clients):
                                    if i not in cfg.Oracle.which_ones and i not in malicious_clients:
                                        clients_to_aggregate.append(client_outputs[i])
                                    else:
                                        clients_to_kick_out.append(i)
                                sim_result["oracle_kicked_clients"].append(clients_to_kick_out)
                        else:
                            # -----------------------------------------
                            #           Group Testing
                            # -----------------------------------------
                            # cross silo only performs GT once
                            cross_silo_gt = r == cfg.GT.group_test_round and n_clients_total == cfg.Sim.n_clients
                            # cross device performs GT every round after the threshold round
                            cross_device_gt = r >= cfg.GT.group_test_round and n_clients_total > cfg.Sim.n_clients
                            
                            if r < cfg.GT.group_test_round:
                                DEC = np.zeros((1, cfg.Sim.n_clients), dtype=np.uint8)
                            elif cross_silo_gt or cross_device_gt:  
                                syndrome = np.matmul(defective[:,sampled_clients_indices], gt.parity_check_matrix.transpose())
                                
                                if noiseless_gt is True:
                                    group_accuracies, prec, rec, f1, loss_value = gt.get_group_accuracies(client_outputs, server, cfg.Data.n_classes)
                                    DEC, LLRoutput = gt.noiseless_group_test(syndrome)
                                    test_values = syndrome
                                else: 
                                    group_accuracies, prec, rec, f1, loss_value = gt.get_group_accuracies(client_outputs, server, cfg.Data.n_classes)
                                    _, _, pca_components, _, _, pca_per_label, _, _ = gt.get_pca_components(client_outputs, server, no_PCA_components, cfg.Data.n_classes)
                                    sim_result["PCA_components_per_label"][monte_carlo_iterr, :, :, :] = pca_per_label
                                    sim_result["PCA_components"][monte_carlo_iterr, :, :] = pca_components
                                    
                                    Neyman_person=True
                                    if ATTACK == 0:
                                        #DEC, LLRoutput = gt.perform_group_test(group_accuracies)
                                        avg_recall = np.mean(rec, axis=1)
                                        test_values, s_scores, d_scores = gt.perform_clustering_and_testing(group_accuracies, pca_components[:,0], cfg.Test.ss_thres)
                                        DEC, LLRoutput, LLRinput, td, nm_est = gt.perform_gt(test_values, Neyman_person)
                                    elif ATTACK == 2:
                                        test_values, s_scores, d_scores = gt.perform_clustering_and_testing( rec[:, src], pca_per_label[src, :, 0], cfg.Test.ss_thres)
                                        DEC, LLRoutput, LLRinput, td, nm_est = gt.perform_gt(test_values, Neyman_person)
                                    
                                malicious_counter[:, sampled_clients_indices] += DEC

                                MD = MD + np.sum(gt.DEC[defective[:, sampled_clients_indices] == 1] == 0)
                                FA = FA + np.sum(gt.DEC[defective[:, sampled_clients_indices] == 0] == 1)
                                if np.sum(DEC) == DEC.shape[1]: 
                                    all_class_malicious = True
                                    
                                sim_result["group_acc"][thres_indx, monte_carlo_iterr, :] = group_accuracies
                                sim_result["group_prec"][thres_indx, monte_carlo_iterr, :, :] = prec
                                sim_result["group_recall"][thres_indx, monte_carlo_iterr, :, :] = rec
                                sim_result["group_f1"][thres_indx, monte_carlo_iterr, :, :] = f1
                                sim_result["DEC"][thres_indx, monte_carlo_iterr, :] = DEC
                                sim_result["LLRO"][thres_indx, monte_carlo_iterr, :] = LLRoutput[0]
                                sim_result["LLRin"][thres_indx, monte_carlo_iterr, :] = LLRinput[0]
                                sim_result["td_used"][thres_indx, monte_carlo_iterr] = td
                                sim_result["nm_est"][thres_indx, monte_carlo_iterr] = nm_est
                                sim_result["syndrome"][thres_indx, monte_carlo_iterr] = syndrome[0]
                                sim_result["test_values"][thres_indx, monte_carlo_iterr] = test_values
                                sim_result["s_scores"][thres_indx, monte_carlo_iterr, :] = s_scores
                                sim_result["d_scores"][thres_indx, monte_carlo_iterr, :] = d_scores
                                
                            # -----------------------------------------
                            #               Aggregation
                            # -----------------------------------------
                            # If all malicious, just use all
                            if all_class_malicious is True:
                                logging.info("All groups are malicious!")
                                clients_to_aggregate = client_outputs
                            else:
                                clients_to_aggregate = [
                                    client_outputs[client_idx]
                                    for client_idx in range(cfg.Sim.n_clients)
                                    if DEC[:, client_idx] == 0
                                ]
                        server_outputs, acc[0, r], cf_matrix = server.run(clients_to_aggregate)
                        true_positives = np.diag(cf_matrix)
                        total_actuals = np.sum(cf_matrix, axis=1)
                        recalls = true_positives / total_actuals
                        balanced_acc = np.mean(recalls)
                        acc[0, r] = balanced_acc

                        sim_result["cf_matrix"][thres_indx, monte_carlo_iterr, r, :, :] = cf_matrix

                        round_end = time.time()
                        if prev_and_cross_sim:
                            logging.info(f"Threshold: {threshold_dec} --- n_mali: {n_malicious} --- MODE: {MODE} --- MC-Iteration: {monte_carlo_iterr} --- Round {r} ---  Time: {round_end - round_start} --- Accuracy: {acc[0,r]} --- prev: {prevalence_sim} --- cross_prop {crossover_probability}")
                        else:
                            logging.info(f"Threshold: {threshold_dec} --- n_mali: {n_malicious} --- MODE: {MODE} --- MC-Iteration: {monte_carlo_iterr} --- Round {r} ---  Time: {round_end - round_start} --- Accuracy: {acc[0,r]}")

                    accuracy[thres_indx, monte_carlo_iterr, :] = acc

            if n_malicious > 0: 
                P_MD[thres_indx] = MD / (n_malicious * cfg.Sim.total_MC_it)
            P_FA[thres_indx] = FA / ((cfg.Sim.n_clients - n_malicious) * cfg.Sim.total_MC_it)

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
            checkpoint_dict["LLRO"] = checkpoint_dict["LLRO"].tolist()
            checkpoint_dict["LLRin"] = checkpoint_dict["LLRin"].tolist()
            checkpoint_dict["td_used"] = checkpoint_dict["td_used"].tolist()
            checkpoint_dict["nm_est"] = checkpoint_dict["nm_est"].tolist()
            checkpoint_dict["syndrome"] = checkpoint_dict["syndrome"].tolist()
            checkpoint_dict["test_values"] = checkpoint_dict["test_values"].tolist()
            checkpoint_dict["malicious_clients"] = checkpoint_dict["malicious_clients"].tolist()
            checkpoint_dict["bsc_channel"] = checkpoint_dict["bsc_channel"].tolist()
            checkpoint_dict["PCA_components_per_label"] = checkpoint_dict["PCA_components_per_label"].tolist()
            checkpoint_dict["PCA_components"] = checkpoint_dict["PCA_components"].tolist()
            checkpoint_dict["s_scores"] = checkpoint_dict["s_scores"].tolist()
            checkpoint_dict["d_scores"] = checkpoint_dict["d_scores"].tolist()
            if cfg.Sim.PCA_simulation is True:
                checkpoint_dict["cosine_similarity_per_label"] = checkpoint_dict["cosine_similarity_per_label"].tolist()
                checkpoint_dict["cosine_similarity_model"] = checkpoint_dict["cosine_similarity_model"].tolist()
                checkpoint_dict["PCA_components"] = checkpoint_dict["PCA_components"].tolist()
                checkpoint_dict["PCA_variance"] = checkpoint_dict["PCA_variance"].tolist()
                checkpoint_dict["PCA_variance_ratio"] = checkpoint_dict["PCA_variance_ratio"].tolist()
                checkpoint_dict["PCA_variance_per_label"] = checkpoint_dict["PCA_variance_per_label"].tolist()
                checkpoint_dict["PCA_variance_ratio_per_label"] = checkpoint_dict["PCA_variance_ratio_per_label"].tolist()
                checkpoint_dict["loss_per_label"] = checkpoint_dict["loss_per_label"].tolist()

            prefix_0 = "./results/estimating_nm/"
            if "mnist" in cfg.Data.data_dir:
                prefix = prefix_0 + "MNIST_"
            elif "cifar" in cfg.Data.data_dir:
                prefix = prefix_0 + "CIFAR10_"
            elif "isic" in cfg.Data.data_dir:
                prefix = prefix_0 + "ISIC_"
            if cfg.Sim.PCA_simulation is True:
                if "mnist" in cfg.Data.data_dir:
                    prefix = f"./PCA_results/PCA_{no_PCA_components}_per_round_non_std_loss/MNIST_"
                elif "cifar" in cfg.Data.data_dir:
                    prefix = f"./PCA_results/PCA_{no_PCA_components}_per_round_non_std_loss/CIFAR10_"
                elif "isic" in cfg.Data.data_dir:
                    prefix = f"./PCA_results/PCA_{no_PCA_components}_per_round_non_std_loss/ISIC_"
            if not prev_and_cross_sim:
                suffix = f"m-{n_malicious},{cfg.Sim.n_clients}_t-{cfg.GT.n_tests}_e-{epochs}_bs-{batch_size}_alpha-{alpha}_totalMC-{cfg.Sim.total_MC_it}_MODE-{MODE}_att-{ATTACK}.txt"
            else:
                suffix = f"m-{n_malicious},{cfg.Sim.n_clients}_t-{cfg.GT.n_tests}_e-{epochs}_bs-{batch_size}_alpha-{alpha}_totalMC-{cfg.Sim.total_MC_it}_MODE-{MODE}_prev-{prevalence_sim}_p-{crossover_probability}_att-{ATTACK}.txt"
            if kick_out_clients is True:
                suffix = "kicked_" + suffix
            sim_title = prefix + suffix

            with open(sim_title, "w") as convert_file:
                convert_file.write(json.dumps(checkpoint_dict))
        if not prev_and_cross_sim:
            zgjimi = f"The simulation for dataset : {cfg_path[16:21]} for {n_malicious} mal_nodes out of {cfg.Sim.n_clients} clients, MODE {MODE}, ATTACK - {ATTACK}, with PCA - {cfg.Sim.PCA_simulation} and kicked - {kick_out_clients} is done!"
        else:
            zgjimi = f"Mismatched: The simulation for {n_malicious} mal_nodes, prev {prevalence_sim} and cross_prop {crossover_probability} is done!" 
        commando_uck = f'echo "{zgjimi}" | mail -s "Simulation done!" marvin.xhemrishi@tum.de'
        coje = subprocess.call(commando_uck, shell = True)
