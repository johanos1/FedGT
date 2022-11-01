import torch
import torch.utils.data as data
import numpy as np
from typing import List, Tuple


def flip_label(label_flips: List[Tuple[int, int]], ds: data.Dataset) -> data.Dataset:
    """flip labels in dataset

    Args:
        label_flips (List[Tuple[int, int]]): list of labels to be flipped and its new value
        ds (data.Dataset): dataset to poison

    Returns:
        data.Dataset: posioned dataset
    """

    target_array = ds.target.cpu().detach().numpy()
    for (old_label, new_label) in label_flips:
        target_array[target_array == old_label] = new_label
    ds.target = torch.from_numpy(target_array)
    return ds


def random_labels(ds: data.Dataset) -> data.Dataset:
    """randomly assign labels in dataset

    Args:
        ds (data.Dataset): dataset to poison

    Returns:
        data.Dataset: posioned dataset
    """
    n_labels = ds.target.shape[0]
    num_classes = len(np.unique(ds.target))
    ds.target = torch.from_numpy(np.random.randint(num_classes, size=(n_labels,)))
    return ds
