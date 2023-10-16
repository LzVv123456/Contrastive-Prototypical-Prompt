import os
import numpy as np
import random
from PIL import Image
from typing import Callable, Optional

from torchvision import datasets
from torch.utils.data import Dataset


class ImageNetR(Dataset):
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        args = None
    ) -> None:
        self.args = args
        self.root = root
        self.data_path = root + '/imagenet-r/'
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = datasets.ImageFolder(self.data_path)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.split_train_test()
        self.set_classes(list(range(len(self.classes))))
    
    def split_train_test(self):
        if self.split == 'train':
            split_txt = self.root + '/imr_train.txt'
        elif self.split == 'test':
            split_txt = self.root + '/imr_test.txt'
        elif self.split == 'val':
            split_txt = self.root + '/imr_val.txt'
        else:
            raise NotImplementedError
        with open(split_txt, 'r') as f:
            lines = f.readlines()
        self.all_imgs = []
        for line in lines:
            img_path = self.data_path + line.strip('\n')
            label = self.class_to_idx[line.split('/')[0]]
            self.all_imgs.append((img_path, label))

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
        img = Image.open(self.img_paths[idx])
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.targets[idx]
        return img, label


if __name__ == '__main__':
    root = '../../Datasets/' 

    train_dataset = ImageNetR(root, split='train')
    val_dataset = ImageNetR(root, split='val')
    test_dataset = ImageNetR(root, split='test')

    print(len(train_dataset), len(val_dataset), len(test_dataset))
    for i in range(100):
        train_dataset.set_classes([i])
        print(len(train_dataset.img_paths), np.unique(np.array(train_dataset.targets)))


