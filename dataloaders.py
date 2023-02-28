import os
import random

import torch
import torch.utils.data as data
from torchvision import datasets, transforms

train_batch_size = 128
test_batch_size = 128
testset_length = 10000


def Omniglot(transform, batch_size, shuffle, num_workers):
    dataset = datasets.Omniglot(root='~/data/Omniglot', background=False, download=True, transform=transform)
    num_total_data = int(len(dataset))
    data_list = list(range(num_total_data))
    random.shuffle(data_list)
    ood_list = data_list[:testset_length]
    oodset = data.Subset(dataset, ood_list)
    oodloader = torch.utils.data.DataLoader(oodset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return oodloader


def KMNIST(transform, batch_size, shuffle, num_workers):
    dataset = datasets.KMNIST(root='~/data/KMNIST', train=False, download=True, transform=transform)
    num_total_data = int(len(dataset))
    data_list = list(range(num_total_data))
    random.shuffle(data_list)
    ood_list = data_list[:testset_length]
    oodset = data.Subset(dataset, ood_list)
    oodloader = torch.utils.data.DataLoader(oodset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return oodloader


def CIFAR10(transform, batch_size, shuffle, num_workers):
    dataset = datasets.CIFAR10(root='~/data/CIFAR10', train=False, download=True, transform=transform)
    num_total_data = int(len(dataset))
    data_list = list(range(num_total_data))
    random.shuffle(data_list)
    ood_list = data_list[:testset_length]
    oodset = data.Subset(dataset, ood_list)
    oodloader = torch.utils.data.DataLoader(oodset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return oodloader


def FashionMNIST(transform, batch_size, shuffle, num_workers):
    dataset = datasets.FashionMNIST(root='~/data/FashionMNIST', train=False, download=True, transform=transform)
    num_total_data = int(len(dataset))
    data_list = list(range(num_total_data))
    random.shuffle(data_list)
    ood_list = data_list[:testset_length]
    oodset = data.Subset(dataset, ood_list)
    oodloader = torch.utils.data.DataLoader(oodset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return oodloader


def SVHN(transform, batch_size, shuffle, num_workers):
    dataset = datasets.SVHN(root='~/data/SVHN', split='test', download=True, transform=transform)
    num_total_data = int(len(dataset))
    data_list = list(range(num_total_data))
    random.shuffle(data_list)
    ood_list = data_list[:testset_length]
    oodset = data.Subset(dataset, ood_list)
    oodloader = torch.utils.data.DataLoader(oodset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return oodloader


# LSUN classroom
def LSUN_CR(train=False, batch_size=None, mean=None, std=None):
    if train:
        print('Warning: Training set for LSUN not available')
    if batch_size is None:
        batch_size = test_batch_size

    transform_base = [transforms.ToTensor()]

    if mean is not None and std is not None:
        transform_base += [transforms.Normalize(mean, std)]

    transform = transforms.Compose([
                                       transforms.Resize(size=(32, 32))
                                   ] + transform_base)
    data_dir = os.path.expanduser('~/data/LSUN')
    num_total_data = 168103
    dataset = datasets.LSUN(data_dir, classes=['classroom_train'], transform=transform)
    data_list = list(range(num_total_data))
    random.shuffle(data_list)
    ood_list = data_list[:testset_length]
    oodset = data.Subset(dataset, ood_list)
    oodloader = torch.utils.data.DataLoader(oodset, batch_size=batch_size, shuffle=False, num_workers=8)
    return oodloader


def TinyImageNet(transform, batch_size, shuffle, num_workers):
    data_path = '~/data/tiny-imagenet/val/'
    oodset = datasets.ImageFolder(data_path, transform=transform)
    oodloader = torch.utils.data.DataLoader(oodset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return oodloader


def Corrupted(data_name, transform, batch_size, shuffle, num_workers):
    if data_name == 'MNIST':
        oodset = datasets.MNIST(root='~/data/MNIST', train=False, download=False, transform=transform)
    elif data_name == 'CIFAR10':
        oodset = datasets.CIFAR10(root='~/data/CIFAR10', train=False, download=False, transform=transform)
    elif data_name == 'CIFAR100':
        oodset = datasets.CIFAR100(root='~/data/CIFAR100', train=False, download=False, transform=transform)
    else:
        raise RuntimeError

    oodloader = torch.utils.data.DataLoader(oodset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return oodloader
