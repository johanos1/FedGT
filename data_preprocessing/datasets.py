import logging
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, EMNIST
from torch.utils.data import random_split, DataLoader, Dataset

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


class Data_Manager:
    def __init__(
        self,
        root,
        dataidxs=None,
        train_transform=None,
        val_transform=None,
        val_size=None,
        train_bs=None,
        test_bs=None,
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.val_size = val_size
        self.train_bs = train_bs
        self.test_bs = test_bs
        self.is_mnist = False

        if "fashionmnist" in self.root:
            dataset = FashionMNIST(self.root, train=True, transform=None, download=True)
            self.is_mnist = True
        elif "emnist" in self.root:
            dataset = EMNIST(self.root, split='byclass', train=True, transform=None, download=True)
            self.is_mnist = True
        elif "mnist" in self.root:
            dataset = MNIST(self.root, train=True, transform=None, download=True)
            self.is_mnist = True
        elif "cifar100" in self.root:
            dataset = CIFAR100(self.root, train=True, transform=None, download=True)
        elif "cifar10" in self.root:
            dataset = CIFAR10(self.root, train=True, transform=None, download=True)
        else:
            logger.info("Dataset not supported")
            return
        
        num_data = len(dataset)
        # Split into validation and training data
        self.train_dataset, self.valid_dataset = random_split(dataset, [num_data - self.val_size, self.val_size])
        
        if "emnist" in self.root:
            self.test_dataset = EMNIST(self.root, split='byclass', train=False, transform=self.val_transform, download=True)
        else:
            self.test_dataset = dataset.__class__(self.root, train=False, transform=self.val_transform, download=True)
        
        self.test_dataset.target = self.test_dataset.targets
        
    
    def get_client_dl(self, dataidxs):
        client_ds = Custom_Dataset(
            self.train_dataset,
            self.train_dataset.indices,
            dataidxs,
            self.train_transform,
            self.is_mnist,
        )

        def _init_fn():
            np.random.seed(12)

        client_dl = DataLoader(client_ds, batch_size=self.train_bs)
        return client_dl

    def get_training_labels(self):
        if self.is_mnist:
            return self.train_dataset.dataset.targets[self.train_dataset.indices]
        else:
            return np.array(self.train_dataset.dataset.targets)[self.train_dataset.indices]

    def get_validation_dl(self):
        val_ds = Custom_Dataset(
            self.valid_dataset,
            self.valid_dataset.indices,
            range(len(self.valid_dataset.indices)),
            self.val_transform,
            self.is_mnist
        )

        def _init_fn():
            np.random.seed(12)

        val_dl = DataLoader(val_ds, batch_size=self.train_bs)
        return val_dl
    

    def get_test_dl(self):
        test_dl = DataLoader(self.test_dataset, batch_size=self.test_bs)
        return test_dl


class Custom_Dataset(Dataset):
    def __init__(self, dataset, train_val_data_idx, dataidxs, transform=None, is_mnist=False):
        self.is_mnist = is_mnist
        self.data = dataset.dataset.data[train_val_data_idx][dataidxs]
        if self.is_mnist:
            self.target = dataset.dataset.targets[train_val_data_idx][dataidxs]
        else:
            self.target = np.array(dataset.dataset.targets)[train_val_data_idx][dataidxs]
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.is_mnist:
            img = Image.fromarray(img.numpy(), mode="L")
        else:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

    def __len__(self):
        return len(self.data)
