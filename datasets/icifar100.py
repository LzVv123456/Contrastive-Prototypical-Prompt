import numpy as np
from typing import Callable, Optional
import torch
from torchvision.datasets import CIFAR100


class iCIFAR100(CIFAR100):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        args = None
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform, train=train, download=download)
        # keep a copy of all data and targets, then let self.data and self.targets indicate current chunk
        self.all_data = self.data
        self.all_targets = self.targets
        self.args = args

    def set_classes(self, target_classes: list, img_id_each_class=None):
        data, targets = [], []
        for label in target_classes:
            current_data = self.all_data[np.array(self.all_targets) == label]
            current_targets = np.full((current_data.shape[0]), label)
            data.append(current_data)
            targets.append(current_targets)
        self.data = np.concatenate(data, axis=0)
        self.targets = np.concatenate(targets, axis=0)

    def get_num_of_class(self, target):
        return np.sum(np.array(self.targets) == target)

    def get_class_chunk(self, target): 
        """
        get start index and end index of the target in self.data
        """
        indexes = (self.targets == target).nonzero()
        start_idx, end_idx = np.min(indexes), np.max(indexes)
        return start_idx, end_idx