import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = (False)  # prevent cuda from optimizing convolutional methods

import numpy as np
import random
import copy
import logging
import os
logging.getLogger('timm').setLevel(logging.WARNING) #ERROR
import time
import json
from itertools import product

# methods
import data_preprocessing.data_loader as dl
import methods.fedavg as fedavg
from models.logistic_regression import logistic_regression
from data_preprocessing.data_poisoning import flip_label, random_labels, permute_labels
from models.resnet import resnet18
from models.eff_net import efficient_net
from defence.group_test import Group_Test
from defence.quantitative_gt import Quantitative_Group_Test
import toml
from dotmap import DotMap

import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method


# 0: permutation attack,
# 1: random labels,
# 2: 1->7 label flip,
# 3: active data-poisoning attack (untargeted)
# 4: active data-poisoning attack (targeted)
# 5: gradient ascent attack
# 6: data-poison and model-poison attack
PERMUTATION_ATTACK = 0
RANDOM_LABELS_ATTACK = 1
LABEL_FLIP_ATTACK = 2
ACTIVE_DATA_POISONING_ATTACK = 3
ACTIVE_DATA_POISONING_TARGETED_ATTACK = 4
DATA_POISON_AND_MODEL_POISON_ATTACK = 6
DATA_POISON_AND_MODEL_POISON_TARGETED_ATTACK = 7

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

if __name__ == "__main__":

    cfg_path = "./cfg_files/cfg_cifar.toml"
    with open(cfg_path, "r") as file:
        cfg = DotMap(toml.load(file))
    gt = Group_Test(cfg.Sim.n_clients, cfg.GT.n_tests)
    parity_check_matrix = gt._get_test_matrix()
    del gt
    sim_result = {}

    # if no key is given, make sure we are using all clients
    n_clients_total = cfg.Sim.n_clients_total if "n_clients_total" in cfg.Sim else cfg.Sim.n_clients
    if n_clients_total == cfg.Sim.n_clients:
        cross_silo_flag = True
    else:
        cross_silo_flag = False
    
    sim_params = list(product(
            cfg.Data.alpha_list,
            cfg.ML.epochs_list,
            cfg.Sim.n_malicious_list,
            cfg.ML.batch_size_list,
            cfg.Sim.MODE_list,
            cfg.Sim.attack_list)
            )

    for (
        alpha,
        epochs,
        n_malicious,
        batch_size,
        MODE,
        ATTACK,
    ) in sim_params:
        
        if (ATTACK in {LABEL_FLIP_ATTACK, 
                       ACTIVE_DATA_POISONING_TARGETED_ATTACK, 
                       DATA_POISON_AND_MODEL_POISON_TARGETED_ATTACK}):
            if "mnist" in cfg.Data.data_dir:
                src = 1
                target = 7
            elif "cifar10" in cfg.Data.data_dir:
                src = 7
                target = 4
            elif "isic" in cfg.Data.data_dir:
                src = 0
                target = 1
        
        no_PCA_components = cfg.PCA.no_comp        

        noiseless_gt = True if MODE == 3 else False
        no_defence = True if MODE == 0 else False
        oracle = True if MODE == 1 else False
        GM_aggregation = True if MODE == 4 else False
        Multi_Krum = True if MODE == 5 else False
        flag_np = True if MODE == 6 else False
        flag_QGT = True if MODE == 7 else False

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
        sim_result["total_MC_it"] = cfg.Sim.total_MC_it
        sim_result["group_acc"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds, cfg.GT.n_tests))
        sim_result["group_prec"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds,cfg.GT.n_tests,cfg.Data.n_classes))
        sim_result["group_recall"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds,cfg.GT.n_tests,cfg.Data.n_classes))
        sim_result["group_f1"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds,cfg.GT.n_tests,cfg.Data.n_classes))
        sim_result["bsc_channel"] = np.zeros((cfg.Sim.total_MC_it, 2, 2))
        sim_result["malicious_clients"] = np.zeros((cfg.Sim.total_MC_it, n_clients_total))
        sim_result["DEC"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds, cfg.Sim.n_clients))
        sim_result["LLRO"] = np.zeros(( cfg.Sim.total_MC_it, cfg.ML.communication_rounds, cfg.Sim.n_clients))
        sim_result["LLRin"] = np.zeros((cfg.Sim.total_MC_it, cfg.ML.communication_rounds, cfg.Sim.n_clients))
        sim_result["sampled_clients"] = np.zeros((cfg.Sim.total_MC_it, cfg.ML.communication_rounds, cfg.Sim.n_clients))
        sim_result["td_used"] = np.zeros((cfg.Sim.total_MC_it, cfg.ML.communication_rounds))
        sim_result["syndrome"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds, cfg.GT.n_tests))
        sim_result["test_values"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds, cfg.GT.n_tests))
        sim_result["accuracy"] = np.zeros((cfg.Sim.total_MC_it, cfg.ML.communication_rounds))
        sim_result["recalls"] = np.zeros((cfg.Sim.total_MC_it, cfg.ML.communication_rounds, cfg.Data.n_classes))
        sim_result["cf_matrix"] = np.zeros((cfg.Sim.total_MC_it,cfg.ML.communication_rounds,cfg.Data.n_classes,cfg.Data.n_classes,))
        sim_result["ss_thres"] = cfg.Test.ss_thres
        no_clusters = int(np.sum(parity_check_matrix, axis = 1).max() + 1) #5 ## HARD_CODED
        sim_result["s_scores"] = np.zeros((cfg.Sim.total_MC_it, cfg.ML.communication_rounds, no_clusters))
        sim_result["d_scores"] = np.zeros((cfg.Sim.total_MC_it, cfg.ML.communication_rounds, no_clusters))
        sim_result["PCA_components_per_label"] = np.zeros((cfg.Sim.total_MC_it, cfg.Data.n_classes, cfg.GT.n_tests, no_PCA_components))

        print(f"Silh_score thresh - {cfg.Test.ss_thres}")

        try:
            set_start_method("spawn")
        except RuntimeError:
            pass

        accuracy = np.zeros((cfg.Sim.total_MC_it, cfg.ML.communication_rounds))
        FA = 0
        MD = 0
        for monte_carlo_iterr in range(cfg.Sim.total_MC_it):
            set_random_seed(monte_carlo_iterr)  # all mc iterations should have same seed for each setting
            # -----------------------------------------
            #           Create attacks
            # -----------------------------------------
            # Assign n_malicious clients
            malicious_clients_pool = np.random.permutation(n_clients_total)
            malicious_clients = malicious_clients_pool[:n_malicious].tolist()
            defective = np.zeros((1, n_clients_total), dtype=np.uint8)
            defective[:, malicious_clients] = 1 
            # if cross-silo, make sure at least one group is benign
            if cross_silo_flag:
                while(True):
                    syndy = np.matmul(defective, parity_check_matrix.transpose())
                    if np.any(syndy[0, :] == 0):
                        break
                    malicious_clients_pool = np.random.permutation(n_clients_total)
                    malicious_clients = malicious_clients_pool[:n_malicious].tolist()
                    defective = np.zeros((1,n_clients_total), dtype=np.uint8)
                    defective[:, malicious_clients] = 1
            print(f"malicious clients: {malicious_clients} \n")
            
            # create attacks
            attacks = list_of_lists = [[] for i in range(n_clients_total)]
            for client in range(n_clients_total):
                if client in malicious_clients:
                    if ATTACK == PERMUTATION_ATTACK:
                        attacks[client].append((permute_labels,))
                    elif ATTACK == RANDOM_LABELS_ATTACK:
                        attacks[client].append((random_labels,))
                    elif ATTACK == LABEL_FLIP_ATTACK:
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

            sim_result["malicious_clients"][monte_carlo_iterr, :] = np.array(defective)
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
            server_args.aggregation = "GM" if GM_aggregation else "Avg"
            server_args.aggregation = "MKrum" if Multi_Krum else "Avg"
            if server_args.aggregation == "MKrum":
                server_args.n_malicious = n_malicious

            server = Server(server_dict, server_args)
            server_outputs = server.start()
            start_round = 0
            print(f"Alpha is equal to {alpha} and aggregation method is {server_args.aggregation}")
     
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
            client_args.attack = ATTACK # to let client know about model poisoning
            
            clients = [Client(client_dict[i], client_args) for i in range(n_clients_total)]

            # -----------------------------------------
            #          Make Group Test Object
            # -----------------------------------------
            if not (oracle or no_defence or GM_aggregation or Multi_Krum):
                P_FA_test = 1e-6 if noiseless_gt else cfg.GT.P_FA
                P_MD_test = 1e-6 if noiseless_gt else cfg.GT.P_MD
                prevalence = n_malicious / cfg.Sim.n_clients 
                beta = cfg.Test.beta if flag_np else None
                if not flag_QGT:
                    gt = Group_Test(
                        n_clients = cfg.Sim.n_clients,
                        n_tests = cfg.GT.n_tests,
                        P_FA_test=P_FA_test,
                        P_MD_test=P_MD_test,
                        prevalence = prevalence,
                        threshold_dec = None,
                        beta=beta
                    )
                    sim_result["bsc_channel"][monte_carlo_iterr, :, :] = gt.ChannelMatrix
                else:
                    gt = Quantitative_Group_Test(cfg.Sim.n_clients, cfg.GT.n_tests, prevalence = prevalence)

                if cross_silo_flag:
                    syndrome = np.matmul(defective, gt.parity_check_matrix.transpose())
                    print(f"Syndrome: {syndrome}")
            
            # -----------------------------------------
            #            Main Loop
            # -----------------------------------------
            # each thread will create a client object containing the client information
            acc = np.zeros((1, cfg.ML.communication_rounds))
            all_class_malicious = False
            
            with mp.Pool(cfg.Sim.n_threads) as p:
                for r in range(start_round, cfg.ML.communication_rounds):
                    round_start = time.time()
                    
                    # set the new server model in the clients
                    update_global_model(clients, server_outputs) 
                    
                    # -----------------------------------------
                    #         Perform local training
                    # -----------------------------------------
                    # randomly sample n_clients from the total clients without replacement
                    sampled_clients_indices = np.sort(np.random.choice(n_clients_total, cfg.Sim.n_clients, replace=False)) # check the sorting
                    
                    # Perform active poisoning if applicable
                    if ATTACK in {ACTIVE_DATA_POISONING_ATTACK, 
                                  ACTIVE_DATA_POISONING_TARGETED_ATTACK,
                                  DATA_POISON_AND_MODEL_POISON_ATTACK}:
                        for i in sampled_clients_indices:
                            if i in malicious_clients:
                                if ATTACK == ACTIVE_DATA_POISONING_TARGETED_ATTACK:
                                    clients[i].active_data_poisoning(src)
                                    num_orig_src = sum(np.array(clients[i].train_dataloader.dataset.target) == src)
                                    num_poisoned_src = sum(np.array(clients[i].poisoned_train_dataloader.dataset.target) == src)
                                    print(f"Client {i}: Original src: {num_orig_src} --- After poison src: {num_poisoned_src}")
                                else:
                                    clients[i].active_data_poisoning()
                    
                    sampled_clients = [clients[i] for i in sampled_clients_indices]
                    # assign clients to threads
                    client_splits = np.array_split(sampled_clients, cfg.Sim.n_threads) 
                    # perform local training
                    client_outputs = p.map(local_training, client_splits) 
                    # flatten the list of lists
                    client_outputs = [c for sublist in client_outputs for c in sublist]
                    # sort the list of dictionaries by the client index
                    client_outputs.sort(key=lambda tup: tup["client_index"])
                    
                    if ATTACK == ACTIVE_DATA_POISONING_TARGETED_ATTACK:
                        for i in malicious_clients:
                            src_cnt = client_outputs[i]["src_cnt"]
                            print(f"client {i} used {src_cnt} samples from src during training")

                    if no_defence or Multi_Krum or GM_aggregation:
                        # if no defence, MKrum or GM keep all clients
                        clients_to_aggregate = client_outputs
    
                    elif oracle:
                        # if oracle, only keep benign clients
                        clients_to_aggregate = []
                        for i in range(cfg.Sim.n_clients):
                            if sampled_clients_indices[i] not in malicious_clients: #Check this for cross-silo!!!!
                                clients_to_aggregate.append(client_outputs[i])
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
                            syndrome = np.matmul(
                                defective[:,np.sort(sampled_clients_indices)], gt.parity_check_matrix.transpose())
                            group_accuracies, prec, rec, f1, loss_value = gt.get_group_accuracies(client_outputs, server, cfg.Data.n_classes)
                            _, _, pca, _, _, pca_per_label, _, _ = gt.get_pca_components(client_outputs, server, no_PCA_components, cfg.Data.n_classes)
                            if noiseless_gt is True:
                                DEC = gt.noiseless_group_test(syndrome) # , LLRoutput
                                test_values = syndrome
                            else: 
                                sim_result["PCA_components_per_label"][monte_carlo_iterr, :, :, :] = pca_per_label
                                # use accuracies
                                ACCURACY_METRIC = {PERMUTATION_ATTACK, 
                                                   RANDOM_LABELS_ATTACK, 
                                                   ACTIVE_DATA_POISONING_ATTACK, 
                                                   DATA_POISON_AND_MODEL_POISON_ATTACK}
                                RECALL_METRIC = {LABEL_FLIP_ATTACK,
                                                 ACTIVE_DATA_POISONING_TARGETED_ATTACK,
                                                 DATA_POISON_AND_MODEL_POISON_TARGETED_ATTACK}
                                
                                if ATTACK in ACCURACY_METRIC:
                                    test_values, s_scores, d_scores = gt.perform_clustering_and_testing(group_accuracies, pca[:, 0], cfg.Test.ss_thres)
                                # use recall
                                elif ATTACK in RECALL_METRIC:
                                    test_values, s_scores, d_scores = gt.perform_clustering_and_testing( rec[:, src], pca_per_label[src, :, 0], cfg.Test.ss_thres)
                                
                                if flag_QGT:
                                    DEC = gt.perform_gt(test_values)
                                else:
                                    DEC, LLRoutput, LLRinput, td, nm_est = gt.perform_gt(test_values, Neyman_person=flag_np)
                            defective_sampled = defective[0, np.sort(sampled_clients_indices)]
                            MD = MD + np.sum(gt.DEC[0, defective_sampled == 1] == 0)
                            FA = FA + np.sum(gt.DEC[0, defective_sampled == 0] == 1)
                            if np.sum(DEC) == DEC.shape[1]:
                                all_class_malicious = True
                            indx = r if cross_device_gt else 0
                            sim_result["group_acc"][monte_carlo_iterr,indx, :] = group_accuracies
                            sim_result["group_prec"][monte_carlo_iterr,indx, :, :] = prec
                            sim_result["group_recall"][monte_carlo_iterr,indx, :, :] = rec
                            sim_result["group_f1"][monte_carlo_iterr,indx, :, :] = f1
                            sim_result["DEC"][monte_carlo_iterr,indx, :] = DEC
                            sim_result["sampled_clients"][monte_carlo_iterr, r, :] = np.sort(sampled_clients_indices)
                            sim_result["test_values"][monte_carlo_iterr,indx] = test_values
                            sim_result["s_scores"][monte_carlo_iterr,indx, :] = s_scores
                            sim_result["d_scores"][monte_carlo_iterr,indx, :] = d_scores
                            sim_result["syndrome"][monte_carlo_iterr,indx] = syndrome[0]   
                            if flag_QGT == False:
                                sim_result["LLRO"][monte_carlo_iterr,indx, :] = LLRoutput[0]
                                sim_result["LLRin"][monte_carlo_iterr,indx, :] = LLRinput[0]
                                sim_result["td_used"][monte_carlo_iterr,indx] = td              
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

                    sim_result["cf_matrix"][monte_carlo_iterr, r, :, :] = cf_matrix
                    sim_result["recalls"][monte_carlo_iterr, r, :] = recalls
                    if ATTACK == 4:
                        print(f"src recall: {recalls[src]}")

                    round_end = time.time()
                    logging.info(f"N_mali: {n_malicious} --- MODE: {MODE} --- MC-Iteration: {monte_carlo_iterr} --- Round {r} ---  Time: {round_end - round_start} --- Accuracy: {acc[0,r]}")

                accuracy[monte_carlo_iterr, :] = acc

        if n_malicious > 0: 
            P_MD = MD / (n_malicious * cfg.Sim.total_MC_it)
        P_FA = FA / ((cfg.Sim.n_clients - n_malicious) * cfg.Sim.total_MC_it)

        # make all nparrays JSON serializable
        checkpoint_dict = copy.deepcopy(sim_result)
        checkpoint_dict["accuracy"] = accuracy.tolist()
        checkpoint_dict["P_MD"] = P_MD
        checkpoint_dict["P_FA"] = P_FA
        checkpoint_dict["group_acc"] = checkpoint_dict["group_acc"].tolist()
        checkpoint_dict["group_prec"] = checkpoint_dict["group_prec"].tolist()
        checkpoint_dict["group_recall"] = checkpoint_dict["group_recall"].tolist()
        checkpoint_dict["group_f1"] = checkpoint_dict["group_f1"].tolist()
        checkpoint_dict["cf_matrix"] = checkpoint_dict["cf_matrix"].tolist()
        checkpoint_dict["DEC"] = checkpoint_dict["DEC"].tolist()
        checkpoint_dict["sampled_clients"] = checkpoint_dict["sampled_clients"].tolist()
        checkpoint_dict["LLRO"] = checkpoint_dict["LLRO"].tolist()
        checkpoint_dict["LLRin"] = checkpoint_dict["LLRin"].tolist()
        checkpoint_dict["td_used"] = checkpoint_dict["td_used"].tolist()
        checkpoint_dict["syndrome"] = checkpoint_dict["syndrome"].tolist()
        checkpoint_dict["test_values"] = checkpoint_dict["test_values"].tolist()
        checkpoint_dict["malicious_clients"] = checkpoint_dict["malicious_clients"].tolist()
        checkpoint_dict["bsc_channel"] = checkpoint_dict["bsc_channel"].tolist()
        checkpoint_dict["PCA_components_per_label"] = checkpoint_dict["PCA_components_per_label"].tolist()
        checkpoint_dict["recalls"] = checkpoint_dict["recalls"].tolist()
        checkpoint_dict["s_scores"] = checkpoint_dict["s_scores"].tolist()
        checkpoint_dict["d_scores"] = checkpoint_dict["d_scores"].tolist()

        prefix_0 = "./results/" #active_attacks
        if "mnist" in cfg.Data.data_dir:
            prefix = prefix_0 + "MNIST_"
        elif "cifar" in cfg.Data.data_dir:
            prefix = prefix_0 + "CIFAR_"
        elif "isic"  in cfg.Data.data_dir:
            prefix = prefix_0 + "ISIC_"
        suffix = f"m-{n_malicious},{cfg.Sim.n_clients}_t-{cfg.GT.n_tests}_e-{epochs}_bs-{batch_size}_alpha-{alpha}_totalMC-{cfg.Sim.total_MC_it}_MODE-{MODE}_att-{ATTACK}.txt"
        sim_title = prefix + suffix
        
        # Ensure the directory exists
        output_dir = os.path.dirname(sim_title)
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Write the JSON data to the file
        with open(sim_title, "w") as convert_file:
            convert_file.write(json.dumps(checkpoint_dict))
