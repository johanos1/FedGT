"""
Dataset Concstruction
Code based on https://github.com/FedML-AI/FedML
"""


import logging
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


class CIFAR_truncated(data.Dataset):
    def __init__(
        self,
        root,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        if "cifar100" in self.root:
            cifar_dataobj = CIFAR100(
                self.root,
                self.train,
                self.transform,
                self.target_transform,
                self.download,
            )
        else:
            cifar_dataobj = CIFAR10(  # Maybe this can offer a validation set
                self.root,
                self.train,
                self.transform,
                self.target_transform,
                self.download,
            )

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class MNIST_truncated(data.Dataset):
    def __init__(
        self,
        root,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        val_size=None,
        # validxs = None,
    ):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.val_size = val_size
        self.validxs = None  # ask about this...

        # only one 'val_size or validxs' should not be None!
        if val_size is None:
            self.data, self.target = self.__build_truncated_dataset__()
        else:
            self.data, self.target, self.validxs = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        """
        Whenever val_size is not None we ask for the validation dataset and indexes (should be the first thing done!)
        Then we output validxs and save it for later in the other function
        we will use the validxs to initialize MNIST without the validation dataset!
        """

        mnist_dataobj = MNIST(
            self.root, self.train, self.transform, self.target_transform, self.download
        )

        if self.val_size is not None:
            num_data_points = len(mnist_dataobj)
            validxs = np.random.permutation(num_data_points)
            validxs = validxs[0 : self.val_size]
            data = mnist_dataobj.data
            target = mnist_dataobj.targets
            data = data[validxs]
            target = target[validxs]
            return data, target, validxs

        data = mnist_dataobj.data
        target = mnist_dataobj.targets

        # if self.validxs is not None:
        #    extra_points = np.setdiff1d(np.arange(num_data_points), np.array(self.validxs))
        #    data = data[extra_points]
        #    target = target[extra_points]

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class FASHION_MNIST_truncated(data.Dataset):
    def __init__(
        self,
        root,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        fmnist_dataobj = FashionMNIST(
            self.root, self.train, self.transform, self.target_transform, self.download
        )

        data = fmnist_dataobj.data
        target = fmnist_dataobj.targets

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
