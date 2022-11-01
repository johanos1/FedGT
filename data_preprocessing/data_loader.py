"""
Federated Dataset Loading and Partitioning
Code based on https://github.com/FedML-AI/FedML
"""

import logging

import numpy as np
from numpy.core.fromnumeric import mean
import torch.utils.data as data
import torchvision.transforms as transforms

from data_preprocessing.datasets import (
    CIFAR_truncated,
    FASHION_MNIST_truncated,
    MNIST_truncated,
)
from data_preprocessing.data_poisoning import flip_label, random_labels

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


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
                transforms.ToPILImage(),
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

    return train_transform, valid_transform


def load_data(datadir):
    train_transform, test_transform = _data_transforms(datadir)
    if "cifar" in datadir:
        dl_obj = CIFAR_truncated
    elif "fashion" in datadir:
        dl_obj = FASHION_MNIST_truncated
    elif "mnist" in datadir:
        dl_obj = MNIST_truncated

    elif "fashionmnist" in datadir:
        pass

    train_ds = dl_obj(datadir, train=True, download=True, transform=train_transform)
    test_ds = dl_obj(datadir, train=False, download=True, transform=test_transform)

    y_train, y_test = train_ds.target, test_ds.target

    return (y_train, y_test)


def partition_data(datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    # load the labels from training and test data sets
    y_train, y_test = load_data(datadir)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
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
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / n_nets)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return class_num, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(datadir, train_bs, test_bs, dataidxs=None, attacks=None):
    train_transform, test_transform = _data_transforms(datadir)
    if "cifar" in datadir:
        dl_obj = CIFAR_truncated

    elif "mnist" in datadir:
        dl_obj = MNIST_truncated

    elif "fashionmnist" in datadir:
        dl_obj = FASHION_MNIST_truncated

    workers = 0
    persist = False

    train_ds = dl_obj(
        datadir, dataidxs=dataidxs, train=True, transform=train_transform, download=True
    )
    test_ds = dl_obj(datadir, train=False, transform=test_transform, download=True)

    # PERFORM ATTACKS BEFORE MAKING DATALOADERS
    if attacks is not None:
        #train_ds = random_labels(train_ds)
        label_flips = [(0, 1), (3, 2)]
        train_ds = flip_label(label_flips, train_ds)

    # Create dataloaders
    train_dl = data.DataLoader(
        dataset=train_ds,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        persistent_workers=persist,
    )
    test_dl = data.DataLoader(
        dataset=test_ds,
        batch_size=test_bs,
        shuffle=False,
        drop_last=True,
        num_workers=workers,
        persistent_workers=persist,
    )

    return train_dl, test_dl


def load_partition_data(
    data_dir, partition_method, partition_alpha, client_number, batch_size
):
    class_num, net_dataidx_map, traindata_cls_counts = partition_data(
        data_dir, partition_method, client_number, partition_alpha
    )

    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(
        data_dir, batch_size, batch_size
    )
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(train_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info(
            "client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num)
        )

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(
            data_dir, batch_size, batch_size, dataidxs, attacks=True
        )

        logging.info(
            "client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d"
            % (client_idx, len(train_data_local), len(test_data_local))
        )
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )
