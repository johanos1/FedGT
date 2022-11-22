"""
Dataset Concstruction
Code based on https://github.com/FedML-AI/FedML
"""


import logging
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torch.utils.data import random_split, DataLoader, Dataset

from data_preprocessing.data_poisoning import flip_label, random_labels

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class CIFAR_Manager:
    def __init__(
        self,
        root,
        dataidxs=None,
        transform=None,
        val_transform=None,
        val_size=None,
        train_bs=None,
        test_bs=None,
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.transform = transform
        self.val_transform = val_transform
        self.val_size = val_size
        self.train_bs = train_bs
        self.test_bs = test_bs
        train = True
        if "cifar100" in self.root:
            cifar_train = CIFAR100(
                self.root,
                train,
                self.transform,
                self.val_transform,
                download=True,
            )
        else:
            cifar_train = CIFAR10(  # Maybe this can offer a validation set
                self.root,
                train,
                self.transform,
                self.val_transform,
                download=True,
            )
        num_data = len(cifar_train)
        self.train_ds, self.valid_ds = random_split(
            cifar_train, [num_data - self.val_size, self.val_size]
        )
        # for some reason, targets for CIFAR is not an array but a list...
        self.train_ds.dataset.targets = np.asarray(self.train_ds.dataset.targets)

    def get_client_dl(self, dataidxs, attacks=None):
        client_ds = Custom_Dataset(self.train_ds, dataidxs, self.transform)

        # Perform attacks before making dataloaders. Data cannot be altered afterwards
        if len(attacks) > 0:
            for attack in attacks:
                f_attack = attack[0]  # first element is the callable
                if len(attack) > 1:  # second element is the optional arguments
                    args = attack[1]
                    client_ds = f_attack(args, client_ds)
                else:
                    client_ds = f_attack(client_ds)

        client_dl = DataLoader(client_ds, batch_size=self.train_bs)
        return client_dl

    def get_training_labels(self):
        return self.train_ds.dataset.targets[self.train_ds.indices]

    def get_validation_dl(self):
        val__ds = Custom_Dataset(
            self.valid_ds, self.valid_ds.indices, self.val_transform
        )
        val_dl = DataLoader(val__ds, batch_size=self.train_bs)
        return val_dl

    def get_test_dl(self):
        train = False
        if "cifar100" in self.root:
            cifar_test = CIFAR100(
                self.root,
                train,
                self.transform,
                self.val_transform,
                download=True,
            )
        else:
            cifar_test = CIFAR10(  # Maybe this can offer a validation set
                self.root,
                train,
                self.transform,
                self.val_transform,
                download=True,
            )

        test_dl = DataLoader(cifar_test, batch_size=self.test_bs)

        return test_dl


class MNIST_Manager:
    def __init__(
        self,
        root,
        dataidxs=None,
        transform=None,
        val_transform=None,
        val_size=None,
        train_bs=None,
        test_bs=None,
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.transform = transform
        self.val_transform = val_transform
        self.val_size = val_size
        self.train_bs = train_bs
        self.test_bs = test_bs

        train = True
        mnist_train = MNIST(self.root, train, self.transform, download=True)
        num_data = len(mnist_train)
        self.train_ds, self.valid_ds = random_split(
            mnist_train, [num_data - self.val_size, self.val_size]
        )

    def get_client_dl(self, dataidxs, attacks=None):
        client_ds = Custom_Dataset(self.train_ds, dataidxs, self.transform)

        # Perform attacks before making dataloaders. Data cannot be altered afterwards
        if len(attacks) > 0:
            for attack in attacks:
                f_attack = attack[0]  # first element is the callable
                if len(attack) > 1:  # second element is the optional arguments
                    args = attack[1]
                    client_ds = f_attack(args, client_ds)
                else:
                    client_ds = f_attack(client_ds)

        client_dl = DataLoader(client_ds, batch_size=self.train_bs)
        return client_dl

    def get_training_labels(self):
        return self.train_ds.dataset.targets[self.train_ds.indices]

    def get_validation_dl(self):
        val__ds = Custom_Dataset(
            self.valid_ds, self.valid_ds.indices, self.val_transform
        )
        val_dl = DataLoader(val__ds, batch_size=self.train_bs)
        return val_dl

    def get_test_dl(self):
        train = False
        fmnist_test = MNIST(self.root, train, self.val_transform, download=True)
        test_dl = DataLoader(fmnist_test, batch_size=self.test_bs)

        return test_dl


class FMNIST_Manager:
    def __init__(
        self,
        root,
        dataidxs=None,
        transform=None,
        val_transform=None,
        val_size=None,
        train_bs=None,
        test_bs=None,
    ):

        self.root = root
        self.dataidxs = dataidxs
        self.transform = transform
        self.val_transform = val_transform
        self.val_size = val_size
        self.train_bs = train_bs
        self.test_bs = test_bs

        train = True
        fmnist_train = FashionMNIST(
            self.root, train, self.transform, self.val_transform, download=True
        )
        num_data = len(fmnist_train)
        self.train_ds, self.valid_ds = random_split(
            fmnist_train, [num_data - self.val_size, self.val_size]
        )

    def get_client_dl(self, dataidxs, attacks=None):
        client_ds = Custom_Dataset(self.train_ds, dataidxs, self.transform)

        # Perform attacks before making dataloaders. Data cannot be altered afterwards
        if len(attacks) > 0:
            for attack in attacks:
                f_attack = attack[0]  # first element is the callable
                if len(attack) > 1:  # second element is the optional arguments
                    args = attack[1]
                    client_ds = f_attack(args, client_ds)
                else:
                    client_ds = f_attack(client_ds)

        client_dl = DataLoader(client_ds, batch_size=self.train_bs)
        return client_dl

    def get_training_labels(self):
        return self.train_ds.dataset.targets[self.train_ds.indices]

    def get_validation_dl(self):
        val__ds = Custom_Dataset(
            self.valid_ds, self.valid_ds.indices, self.val_transform
        )
        val_dl = DataLoader(val__ds, batch_size=self.train_bs)
        return val_dl

    def get_test_dl(self):
        train = False
        fmnist_test = FashionMNIST(self.root, train, self.val_transform, download=True)
        test_dl = DataLoader(fmnist_test, batch_size=self.test_bs)

        return test_dl


class Custom_Dataset(Dataset):
    def __init__(self, ds, dataidxs, transform):
        self.data = ds.dataset.data[dataidxs]
        self.target = ds.dataset.targets[dataidxs]
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # to be consistent with CIFAR to return a PIL Image
        if isinstance(img, np.ndarray) is False:
            img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
