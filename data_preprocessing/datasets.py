import logging
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, EMNIST
from torch.utils.data import random_split, DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

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

class TabularDataset(Dataset):
    """Tabular dataset."""

    def __init__(self, X, y):
        """Initializes instance of class TabularDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        self.data = X
        self.targets= y

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        X = np.float32(self.data[idx])
        y = np.float32(self.targets[idx])
        return [X, y]

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
            dataset = EMNIST(self.root, split='digits', train=True, transform=None, download=True)
            self.is_mnist = True
        elif "mnist" in self.root:
            dataset = MNIST(self.root, train=True, transform=None, download=True)
            self.is_mnist = True
        elif "cifar100" in self.root:
            dataset = CIFAR100(self.root, train=True, transform=None, download=True)
        elif "cifar10" in self.root:
            dataset = CIFAR10(self.root, train=True, transform=None, download=True)
        elif "adult" in self.root:
            path = "./data/adult"
            dataset_name = "adult"

            column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                    "hours-per-week", "native-country", "income"]
            df_train = pd.read_csv(f"{path}/{dataset_name}.data", names=column_names)
            
            
            df_test = pd.read_csv(f"{path}/{dataset_name}.test", names=column_names, header=0)
            df_test['income'] = df_test['income'].str.replace('.', '', regex=False)

            def preprocess(df, train_encoder=None, train_scaler=None, feature_names=None):
                df = df.replace(' ?', np.nan)
                df = df.dropna()
                X, y = df.iloc[:, :-1], df.iloc[:, -1]

                categorical_features = [col for col in X.columns if X[col].dtype == 'object']
                numerical_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

                if train_encoder is not None:
                    onehot_encoder = train_encoder
                    X_categorical = onehot_encoder.transform(X[categorical_features])
                    scaler = train_scaler
                    X_numerical = scaler.transform(X[numerical_features])
                else:
                    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    X_categorical = onehot_encoder.fit_transform(X[categorical_features])
                    scaler = StandardScaler()
                    X_numerical = scaler.fit_transform(X[numerical_features])

                X = np.hstack([X_numerical, X_categorical])
                
                # If feature_names is provided (for test set), ensure column consistency
                if feature_names is not None:
                    missing_cols = set(feature_names) - set(onehot_encoder.get_feature_names_out())
                    for col in missing_cols:
                        # Find the index to insert the missing column at the correct position
                        index = list(feature_names).index(col)
                        X = np.insert(X, index, 0, axis=1)  # Insert zeros for missing column

                # label encode the target variable to have the classes 0 and 1
                y = LabelEncoder().fit_transform(y)
                return X, y, train_encoder, train_scaler, onehot_encoder.get_feature_names_out()

            X_train, y_train, train_encoder, train_scaler, feature_names_train = preprocess(df_train)
            X_test, y_test, _, _, _ = preprocess(df_test, train_encoder, train_scaler, feature_names_train)
            dataset = TabularDataset(X_train, y_train)
            num_data = len(dataset)
            train_ds, val_ds = random_split(dataset, [num_data - self.val_size, self.val_size])
            
            train_indx = np.asarray(train_ds.indices)
            self.train_dataset = TabularDataset(train_ds.dataset.data[train_indx], train_ds.dataset.targets[train_indx])
            val_indx = np.asarray(val_ds.indices)
            self.valid_dataset = TabularDataset(val_ds.dataset.data[val_indx], val_ds.dataset.targets[val_indx])
            self.test_dataset = TabularDataset(X_test, y_test)
            self.test_dataset.target = self.test_dataset.targets
            return
        
        
        num_data = len(dataset)
        # Split into validation and training data
        self.train_dataset, self.valid_dataset = random_split(dataset, [num_data - self.val_size, self.val_size])
        
        if "emnist" in self.root:
            self.test_dataset = EMNIST(self.root, split='digits', train=False, transform=self.val_transform, download=True)
        elif "adult" in self.root:
            pass
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
        if "adult" in self.root:
            return self.train_dataset.targets
            
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
