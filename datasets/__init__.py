from .icifar100 import iCIFAR100
from .imagenet_sub import ImageNetSub
from .imagenet_r import ImageNetR
from .multidatasets import MultiDatasets
from .proto import ProtoDataset

import os
import utils
from PIL import Image
from torchvision import datasets, transforms


def prepare_dataset(args):
    train_proto_dataset, gen_proto_dataset, test_dataset = \
    None, None, None

    if args.dataset == 'cifar100':
        # set transformations
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        if args.pretrain_method in ['21k', '1k']:
            normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

        # transform for training prototype
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=[0.8, 1.0], interpolation=Image.BICUBIC),
            # transforms.autoaugment.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])

        # transform for others
        test_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            normalize,
        ])

        train_proto_dataset = iCIFAR100(args.data_path, transform=train_transform, train=True, download=True, args=args)
        gen_proto_dataset = iCIFAR100(args.data_path, transform=test_transform, train=True, args=args)
        test_dataset = iCIFAR100(args.data_path, transform=test_transform, train=False, args=args)

    elif args.dataset == 'imagenet_sub':
        # set transformations
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        if args.pretrain_method in ['21k', '1k']:
            normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=[0.8, 1.0], interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            normalize,
        ])

        train_proto_dataset = ImageNetSub(args.data_path, transform=train_transform, train=True, args=args)
        gen_proto_dataset = ImageNetSub(args.data_path, transform=test_transform, train=True, args=args)
        test_dataset = ImageNetSub(args.data_path, transform=test_transform, train=False, args=args)


    elif args.dataset == 'imagenet_r':
        # set transformations
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        if args.pretrain_method in ['21k', '1k']:
            normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=[0.8, 1.0], interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            normalize,
        ])

        train_proto_dataset = ImageNetR(args.data_path, transform=train_transform, split='train', args=args)
        gen_proto_dataset = ImageNetR(args.data_path, transform=test_transform, split='train', args=args)
        test_dataset = ImageNetR(args.data_path, transform=test_transform, split='test', args=args)


    elif args.dataset == '5datasets':
        all_train_transforms, all_test_transforms = [], []

        mean_datasets = {
            'CIFAR10': [x/255 for x in [125.3,123.0,113.9]],
            'notMNIST': (0.4254,),
            'MNIST': (0.1,) ,
            'SVHN':[0.4377,0.4438,0.4728] ,
            'FashionMNIST': (0.2190,)
        }

        std_datasets = {
            'CIFAR10': [x/255 for x in [63.0,62.1,66.7]],
            'notMNIST': (0.4501,),
            'MNIST': (0.2752,),
            'SVHN': [0.198,0.201,0.197],
            'FashionMNIST': (0.3318,)
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
        for dataset_name, _ in mean_datasets.items():
            # normalize
            if args.pretrain_method in ['21k', '1k']:
                normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            else:
                # normalize
                normalize = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean_datasets[dataset_name], std_datasets[dataset_name]),
                ])

            if 'MNIST' in dataset_name:
                # transform for training prototype
                train_transform = transforms.Compose([
                    transforms.Pad(padding=2, fill=0),
                    transforms.RandomResizedCrop(224, scale=[0.8, 1.0], interpolation=Image.BICUBIC),
                    flip_and_color_jitter,
                    utils.GaussianBlur(0.1),
                    utils.Solarization(0.2),
                    normalize,
                ])
                # transform for others
                test_transform = transforms.Compose([
                    transforms.Pad(padding=2, fill=0),
                    transforms.Resize(224, interpolation=Image.BICUBIC),
                    normalize,
                ])

            elif dataset_name == 'svhn':
                # transform for training prototype
                train_transform = transforms.Compose([
                    transforms.Resize(256, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.autoaugment.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.SVHN),
                    transforms.RandomResizedCrop(224, scale=[0.8, 1.0], interpolation=3),
                    flip_and_color_jitter,
                    utils.GaussianBlur(0.1),
                    utils.Solarization(0.2),
                    normalize,
                ])

                # transform for others
                test_transform = transforms.Compose([
                    transforms.Resize(256, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(224),
                    normalize,
                ])

            else:
                # transform for training prototype
                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=[0.8, 1.0], interpolation=Image.BICUBIC),
                    flip_and_color_jitter,
                    utils.GaussianBlur(0.1),
                    utils.Solarization(0.2),
                    normalize,
                ])
                # transform for others
                test_transform = transforms.Compose([
                    transforms.Resize(224, interpolation=Image.BICUBIC),
                    normalize,
                ])

            all_train_transforms.append(train_transform)
            all_test_transforms.append(test_transform)
                
        train_proto_dataset = MultiDatasets(args.data_path, all_transforms=all_train_transforms, train=True, args=args)
        gen_proto_dataset = MultiDatasets(args.data_path, all_transforms=all_test_transforms, train=True, args=args)
        test_dataset = MultiDatasets(args.data_path, all_transforms=all_test_transforms, train=False, args=args)  

    else:
        raise NotImplementedError

    return train_proto_dataset, gen_proto_dataset, test_dataset