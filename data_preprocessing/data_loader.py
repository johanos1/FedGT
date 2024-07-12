"""
Federated Dataset Loading and Partitioning
Code based on https://github.com/FedML-AI/FedML
"""

import sys
sys.path.append('/nas/lnt/ga53rum/packages/')
import logging

import numpy as np
import torchvision.transforms as transforms
import random
import albumentations

from data_preprocessing.datasets import Data_Manager


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def record_net_data_stats(server_val_dl, server_test_dl, client_dict_dl):

    val_unique, val_counts = np.unique(np.array(server_val_dl.dataset.target), return_counts=True)
    tmp = {val_unique[i]: val_counts[i] for i in range(len(val_unique))}
    logging.info(f"\nServer validation set: {str(tmp)}")

    test_unique, test_counts = np.unique(np.array(server_test_dl.dataset.target), return_counts=True)
    tmp = {test_unique[i]: test_counts[i] for i in range(len(test_unique))}
    logging.info(f"Server test set: {str(tmp)}\n")

    for (i, dl) in client_dict_dl.items():
        unq, unq_cnt = np.unique(dl.dataset.target, return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        logging.info(f"Client {i} statistics: {tmp}")


def _data_transforms(datadir):
    if "cifar" in datadir:
        if "cifar100" in datadir:
            CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
            CIFAR_STD = [0.2673, 0.2564, 0.2762]
        else:
            CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
            CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
    elif "fashion" in datadir:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),  # first, convert image to PyTorch tensor
            ]
        )
        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),  # first, convert image to PyTorch tensor
            ]
        )

    elif "mnist" in datadir:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),  # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
            ]
        )
        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),  # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
            ]
        )
    
    elif "isic" in datadir:
        sz = 200
        train_transform = albumentations.Compose(
                [
                    albumentations.RandomScale(0.07),
                    albumentations.Rotate(50),
                    albumentations.RandomBrightnessContrast(0.15, 0.1),
                    albumentations.Flip(p=0.5),
                    albumentations.Affine(shear=0.1),
                    albumentations.RandomCrop(sz, sz),
                    albumentations.CoarseDropout(random.randint(1, 8), 16, 16),
                    albumentations.Normalize(always_apply=True),
                ]
            )
        valid_transform = albumentations.Compose(
                [
                    albumentations.CenterCrop(sz, sz),
                    albumentations.Normalize(always_apply=True),
                ]
            )

    return train_transform, valid_transform


def partition_data(data_obj, partition, n_nets, alpha):
    """
    Inputs:
        datadir -> mnist, fashion, cifar
        partition -> homo or hetero
        n_nets -> number of devices
        alpha -> hetero parameter
    Outputs:

    """
    logging.info("*********partition data***************")
    # load the labels from training and test data sets
    y_train = data_obj.get_training_labels()
    n_train = len(y_train)
    class_num = len(np.unique(y_train))

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = class_num
        N = n_train
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    return class_num, net_dataidx_map

    
def get_data_object(datadir, val_size, batch_size):
    train_transform, val_test_transform = _data_transforms(datadir)
    test_bs_marvin = 1
    dl_obj = Data_Manager(
        datadir,
        None,
        train_transform,
        val_test_transform,
        val_size,
        batch_size,
        test_bs=test_bs_marvin, #batch_size,
    )

    return dl_obj


def load_partition_data(
    data_dir,
    partition_method,
    partition_alpha,
    client_number,
    batch_size,
    attacks,
    val_size,
):

    data_obj = get_data_object(data_dir, val_size, batch_size)

    # create data for server
    server_val_dl = data_obj.get_validation_dl()
    val_data_num = len(server_val_dl.dataset)

    # get local dataset
    client_data_num = dict()
    client_dl_dict = dict()
    
    if data_obj.is_isic:
        server_test_dl = data_obj.get_test_dl_isic()
        net_dataidx_map = data_obj.splitISIC2019(client_number)
        class_num = len(np.unique(server_test_dl.dataset.target))
    else:
        server_test_dl = data_obj.get_test_dl()
        # Start looking at data for clients
        class_num, net_dataidx_map = partition_data(data_obj, partition_method, client_number, partition_alpha)
    test_data_num = len(server_test_dl.dataset)
    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        client_data_num[client_idx] = local_data_num

        client_dl = data_obj.get_client_dl(dataidxs, attacks[client_idx])
        client_dl_dict[client_idx] = client_dl

        logging.info(
            "client_idx = %d, local_sample_number = %d, batch_num = %d" % (client_idx, local_data_num, len(client_dl))
        )

    record_net_data_stats(server_val_dl, server_test_dl, client_dl_dict)

    return (
        val_data_num,
        test_data_num,
        server_val_dl,
        server_test_dl,
        client_data_num,
        client_dl_dict,
        class_num,
    )
