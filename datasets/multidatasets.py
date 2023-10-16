import os
from pickle import NONE
import numpy as np
import urllib.request
from PIL import Image
from typing import Optional

from torchvision import datasets
from torch.utils.data import Dataset


class MultiDatasets(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        all_transforms: Optional[list] = None,
        all_target_transforms: Optional[list] = None,
        args = None
    ) -> None:

        self.root = root
        self.args = args
        self.all_transforms = all_transforms
        self.all_target_transforms = all_target_transforms
        self.transform = None
        self.target_transform = None

        self.all_data = []
        self.all_target = []
        self.data = None
        self.targets = None
        
        # cifar10
        print('preparing cifar10 ...')
        cifar10_data = datasets.CIFAR10(self.root, train=train, download=True).data
        cifar10_label = datasets.CIFAR10(self.root, train=train).targets
        self.all_data.append(cifar10_data)
        self.all_target.append(cifar10_label)

        # not_mnist
        print('preparing not_mnist ...')
        notmnist_data = notMNIST(self.root, train=train, download=True).data
        notmnist_label = notMNIST(self.root, train=train).targets
        self.all_data.append(notmnist_data)
        self.all_target.append(notmnist_label)

        # mnistRGB
        print('preparing mnist ...')
        mnist_data = datasets.MNIST(self.root, train=train, download=True).data
        mnist_label = datasets.MNIST(self.root, train=train).targets
        self.all_data.append(mnist_data)
        self.all_target.append(mnist_label)

        # svhn
        print('preparing svhn ...')
        if train:
            svhn_data = datasets.SVHN(self.root, split='train', download=True).data
            svhn_label = datasets.SVHN(self.root, split='train').labels
        else:
            svhn_data = datasets.SVHN(self.root, split='test', download=True).data
            svhn_label = datasets.SVHN(self.root, split='test').labels
        self.all_data.append(svhn_data)
        self.all_target.append(svhn_label)

        # fashion_mnist
        print('preparing fashion_mnist ...')
        fmnist_data = datasets.FashionMNIST(self.root, train=train, download=True).data
        fmnist_label = datasets.FashionMNIST(self.root, train=train).targets
        self.all_data.append(fmnist_data)
        self.all_target.append(fmnist_label)
    

    def set_classes(self, target_classes:list):
        dataset_idx = np.unique(np.array(target_classes) // 10)
        if len(dataset_idx) == 1:
            assert len(dataset_idx) == 1
            dataset_idx = int(dataset_idx)
            cur_dataset_data = np.array(self.all_data[dataset_idx])
            cur_dataset_label = np.array(self.all_target[dataset_idx])
            assert len(cur_dataset_data) == len(cur_dataset_label)
            self.data = []
            self.targets = [] 
            self.dataset_label = None 
            for class_idx in target_classes:
                class_data = cur_dataset_data[cur_dataset_label==(class_idx % 10)]
                class_label = np.full((class_data.shape[0]), class_idx)
                self.data.append(class_data)
                self.targets.append(class_label)
            self.data = np.concatenate(self.data, axis=0)
            self.targets = np.concatenate(self.targets, axis=0)

            if self.all_transforms is not None:
                self.transform = self.all_transforms[dataset_idx]
            if self.all_target_transforms is not None:
                self.target_transform = self.all_target_transforms[dataset_idx]
        else:
            self.data = []
            self.targets = [] 
            self.dataset_label = []
            for idx in dataset_idx:
                cur_dataset_data = self.all_data[int(idx)]
                cur_dataset_label = self.all_target[int(idx)]
                for i in range(len(cur_dataset_label)):
                     self.data.append(cur_dataset_data[i])
                     self.targets.append(cur_dataset_label[i] + 10 * idx)
                     self.dataset_label.append(int(idx))


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
        return len(self.targets)

    def __getitem__(self, idx):
        img_array = self.data[idx]
        target = self.targets[idx]

        if np.shape(img_array)[0] == 3:
            img_array = np.transpose(img_array, (1, 2, 0))
        img = Image.fromarray(np.array(img_array)).convert('RGB')

        if self.dataset_label is not None:
            self.transform = self.all_transforms[self.dataset_label[idx]]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class notMNIST(Dataset):

    def __init__(self, root, train=True, task_num=None, num_samples_per_class=None, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        self.url = "https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/master/data/notMNIST.zip"
        self.filename = 'notMNIST.zip'
        fpath = os.path.join(root, self.filename)
        if not os.path.isfile(fpath):
            if not download:
               raise RuntimeError('Dataset not found. You can use download=True to download it')
            else:
                print('Downloading from '+self.url)
                download_url(self.url, root, filename=self.filename)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()

        if self.train:
            fpath = os.path.join(root, 'notMNIST', 'Train')
        else:
            fpath = os.path.join(root, 'notMNIST', 'Test')

        X, Y = [], []
        folders = os.listdir(fpath)

        for folder in folders:
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    X.append(np.array(Image.open(img_path).convert('RGB')))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    print("File {}/{} is broken".format(folder, ims))
        self.data = np.array(X)
        self.targets = Y

        self.num_classes = len(set(self.targets))

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def download(self):
        """Download the notMNIST data if it doesn't exist in processed_folder already."""

        import errno
        root = os.path.expanduser(self.root)

        fpath = os.path.join(root, self.filename)

        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        urllib.request.urlretrieve(self.url, fpath)

        import zipfile
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(root)
        zip_ref.close()


if __name__ == '__main__':

    from torchvision import datasets, transforms
    import sys
    sys.path.append('../')
    import utils

    all_train_transforms, all_test_transforms = [], []

    mean_datasets = {
        'CIFAR10': [x/255 for x in [125.3,123.0,113.9]],
        'notMNIST': (0.4254, 0.4254, 0.4254),
        'MNIST': (0.1, 0.1, 0.1) ,
        'SVHN':[0.4377,0.4438,0.4728] ,
        'FashionMNIST': (0.2190, 0.2190, 0.2190),
    }
    std_datasets = {
        'CIFAR10': [x/255 for x in [63.0,62.1,66.7]],
        'notMNIST': (0.4501, 0.4501, 0.4501),
        'MNIST': (0.2752, 0.2752, 0.2752),
        'SVHN': [0.198,0.201,0.197],
        'FashionMNIST': (0.3318, 0.3318, 0.3318)
    }

    # set transformations
    flip_and_color_jitter = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
    ])

    # save all transforms
    for dataset_name in mean_datasets.keys():
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_datasets[dataset_name], std_datasets[dataset_name]),
        ])

        # transform for training prototype
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transform for others
        test_transform = transforms.Compose([
            transforms.Resize(224, interpolation=3),
            normalize,
        ])
        all_train_transforms.append(train_transform)
        all_test_transforms.append(test_transform)
                

    multi_datasets_train = MultiDatasets(root='../../Datasets', train=True, all_transforms=all_train_transforms) 
    target_classes = list(range(30, 40))
    multi_datasets_train.set_classes(target_classes=target_classes)


    for img, label in multi_datasets_train:
        print(np.shape(img), type(img))
        exit()
