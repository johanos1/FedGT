import logging
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torch.utils.data import random_split, DataLoader, Dataset
from data_preprocessing.download_isic import download_isic
from sklearn.model_selection import train_test_split

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
        self.is_isic = False

        if "fashionmnist" in self.root:
            dataObj = FashionMNIST
            self.is_mnist = True
        elif "mnist" in self.root:
            dataObj = MNIST
            self.is_mnist = True
        elif "cifar100" in self.root:
            dataObj = CIFAR100
        elif "cifar10" in self.root:
            dataObj = CIFAR10
        elif "isic2019" in self.root:
            self.is_isic = True
        else:
            logger.info("Dataset not supported")
            return            

        if "isic2019" in self.root:
            # make the ISIC metadata into a dataset
            gt_frame, x, source_folder = download_isic()
            
            y_cols = ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC","UNK"]
            y = np.argmax(gt_frame[y_cols].values, axis=1)
            indices = gt_frame.index.tolist()
            
            n_data = len(y)
            n_train_data = np.floor(0.85 * n_data)
            test_val_ratio = (n_data - n_train_data)/n_data
            train_indices, X_temp, y_train, y_temp = train_test_split(indices, y, test_size=test_val_ratio, random_state=12)
            
            n_temp_data = n_data - n_train_data
            n_test_ratio = (n_temp_data - self.val_size)/n_temp_data
            val_indices, test_indices, y_val, y_test = train_test_split(X_temp, y_temp, test_size=n_test_ratio, random_state=12)

            self.train_dataset = ISICDataset(x, y, train_indices, source_folder)
            self.valid_dataset = ISICDataset(x, y, val_indices, source_folder)
            self.test_dataset = ISICDataset(x, y, test_indices, source_folder)
            
        else:
            dataset = dataObj(self.root, train=True, transform=None, download=True)
            num_data = len(dataset)
            # Split into validation and training data
            self.train_dataset, self.valid_dataset = random_split(dataset, [num_data - self.val_size, self.val_size])
            self.test_dataset = dataObj(self.root, train=False, transform=self.val_transform, download=True)
            self.test_dataset.target = self.test_dataset.targets

    def splitISIC2019(self, n_target):
        # get all the indices that are used for training
        x = self.train_dataset.x.iloc[self.train_dataset.indices]
        y = self.train_dataset.y[self.train_dataset.indices]
        
        # Step 1: split training indices into centers
        x_grouped = x.groupby('dataset').apply(lambda p: p.index.tolist()).tolist()
        n_groups = len(x_grouped)
        
        # Step 2: split training data from largest centers into chunks until we have n_target groups
        while n_groups < n_target:
            group_counts = [len(x_grouped[i]) for i in range(n_groups)]
            largest_group = np.argmax(group_counts)
            
            lst = x_grouped[largest_group]
            mid_idx = len(lst) // 2
            first_half = lst[:mid_idx]
            second_half = lst[mid_idx:]
            x_grouped[largest_group] = first_half
            x_grouped.append(second_half)
            n_groups = len(x_grouped)
        
        # convert to indices inside the training sample indices
        x_index_list = x.index.tolist()
        for j in range(n_groups):
            x_grouped[j] = [i for i, item in enumerate(x_index_list) if item in x_grouped[j]]
        return x_grouped
        
    
    def get_client_dl(self, dataidxs, attacks=None):
        client_ds = Custom_Dataset(
            self.train_dataset,
            self.train_dataset.indices,
            dataidxs,
            self.train_transform,
            self.is_mnist,
            self.is_isic
        )

        # Perform attacks before making dataloaders. Data cannot be altered afterwards
        if len(attacks) > 0:
            for attack in attacks:
                f_attack = attack[0]  # first element is the callable
                if len(attack) > 1:  # second element is the optional arguments
                    args = attack[1]
                    client_ds = f_attack(args, client_ds)
                else:
                    client_ds = f_attack(client_ds)

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
            self.is_mnist,
            self.is_isic
        )

        def _init_fn():
            np.random.seed(12)

        val_dl = DataLoader(val_ds, batch_size=self.train_bs)
        return val_dl
    
    def get_test_dl_isic(self):
        test_ds = Custom_Dataset(
            self.test_dataset,
            self.test_dataset.indices,
            range(len(self.test_dataset.indices)),
            self.val_transform,
            self.is_mnist,
            self.is_isic
        )
        test_dl = DataLoader(test_ds, batch_size=self.train_bs)
        return test_dl

    def get_test_dl(self):
        test_dl = DataLoader(self.test_dataset, batch_size=self.test_bs)
        return test_dl


class ISICDataset(Dataset):
    def __init__(self, x, y, indices, root_dir):
        self.y = y
        self.x = x
        self.indices = indices
        self.root_dir = root_dir

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Load image
        img_path = self.root_dir + '/' + self.x['image'][idx] + '.jpg'
        img = Image.open(img_path).convert('RGB')  # Convert to RGB (in case some images are grayscale)

        return img, self.y[idx]


class Custom_Dataset(Dataset):
    def __init__(self, dataset, train_val_data_idx, dataidxs, transform=None, is_mnist=False, is_isic=False):
        self.is_mnist = is_mnist
        self.is_isic = is_isic
        if self.is_isic:
            self.data = []
            self.target = []
        
            data_indices = [train_val_data_idx[i] for i in dataidxs]
            
            for idx in data_indices:
                image, label = dataset[idx]
                self.data.append(image)
                self.target.append(label)
        else:
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
        elif self.is_isic:
            img = np.array(img)
        else:
            img = Image.fromarray(img)

        if self.transform is not None:
            if self.is_isic:
                img = self.transform(image=img)['image']
                img = np.transpose(img, (2, 0, 1))
            else:
                img = self.transform(img)
        
        return img, target

    def __len__(self):
        return len(self.data)
