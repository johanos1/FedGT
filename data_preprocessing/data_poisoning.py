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

    target_array = np.array(ds.target)
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
    if ds.is_isic:
            num_classes = 8
    else:
        num_classes = 10
    ds.target = torch.from_numpy(np.random.randint(num_classes, size=(n_labels,)))
    return ds


def permute_labels(ds: data.Dataset) -> data.Dataset:
    """permute the labels in dataset

    Args:
        ds (data.Dataset): dataset to poison

    Returns:
        data.Dataset: posioned dataset
    """
    if ds.is_isic:
        num_classes = 8
    else:
        num_classes = 10
    
    target_array = np.array(ds.target)
    target_array = (target_array + 1) % num_classes
    ds.target = torch.from_numpy(target_array)
    return ds
