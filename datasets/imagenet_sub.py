import os
import numpy as np
from PIL import Image
from typing import Callable, Optional

from torchvision import datasets
from torch.utils.data import Dataset


class ImageNetSub(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        args = None
    ) -> None:
        self.args = args
        if train:
            self.data_path = root + '/seed_1993_subset_100_imagenet/data/train'
        else:
            self.data_path = root + '/seed_1993_subset_100_imagenet/data/val'
        self.transform = transform
        self.target_transform = target_transform

        self.dataset = datasets.ImageFolder(self.data_path)
        self.all_imgs = self.dataset.imgs
        self.all_classes = self.dataset.classes
        self.all_class_to_idx = self.dataset.class_to_idx
        # split img path and target
        self.set_classes(list(range(len(self.all_classes))))

    def set_classes(self, target_classes: list, img_id_each_class=None):
        self.img_paths, self.targets = [], []
        for path, class_idx in self.all_imgs:
            if class_idx in target_classes:
                self.img_paths.append(path)
                self.targets.append(class_idx)
            
    def get_num_of_class(self, target):
        return np.sum(np.array(self.targets) == target)

    def get_class_chunk(self, target): 
        """
        get start index and end index of the target in self.data
        """
        indexes = (self.targets == target).nonzero()
        start_idx, end_idx = np.min(indexes), np.max(indexes)
        return start_idx, end_idx

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        with open(self.img_paths[idx], "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.targets[idx]
        return img, label


if __name__ == '__main__':
    root = '../../Datasets/' 
    train_dataset = ImageNetSub(root, train=True)
    val_dataset = ImageNetSub(root, train=False)

    print(len(train_dataset.img_paths))
    train_dataset.set_classes([99])

    for i in range(100):
        val_dataset.set_classes([i])
        print(len(val_dataset.img_paths), np.unique(np.array(val_dataset.targets)))
