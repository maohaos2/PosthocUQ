from __future__ import print_function

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import dataloaders as loaders
import models
import preproc as pre
from metrics import compute_total_entropy, compute_max_prob, compute_differential_entropy, compute_mutual_information, \
    compute_precision
from utils import ROC_OOD, ROC_Selective, convert_to_rgb

parser = argparse.ArgumentParser(description='Meta model Evaluation')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--base_model', default="VGG16_BaseModel", type=str, help='model type (default: LeNet)')
parser.add_argument('--base_epoch', default=200, type=int, help='total epochs to train base model')
parser.add_argument('--meta_model', default="VGG16_MetaModel_combine", type=str, help='model type (default: LeNet)')
parser.add_argument('--fea_dim', default=[16384, 8192, 4096, 2048, 512])
parser.add_argument('--name', default='CIFAR10_OOD', type=str, help='name of run')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='name of run')
parser.add_argument('--seed_trail', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=20, type=int, help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()


if args.dataset == 'MNIST':
    args.fea_dim = [6 * 14 * 14, 5 * 5 * 16]
else:
    args.fea_dim = [16384, 8192, 4096, 2048, 512]

if use_cuda:
    torch.manual_seed(args.seed_trail)
    torch.cuda.manual_seed_all(args.seed_trail)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed_trail)
    random.seed(args.seed_trail)
    os.environ['PYTHONHASHSEED'] = str(args.seed_trail)

'''
Processing data
'''
print('==> Preparing data..')
# Noisy validation set for OOD
if args.dataset == 'MNIST':
    transform_noise = transforms.Compose([
        transforms.Resize(32),
        transforms.Lambda(convert_to_rgb),
        transforms.ToTensor(),
        pre.PermutationNoise(),
        pre.GaussianFilter(),
        pre.ContrastRescaling(),
    ])
else:
    transform_noise = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        pre.PermutationNoise(),
        pre.GaussianFilter(),
        pre.ContrastRescaling(),
    ])

transform_ood_color = transforms.Compose([
    transforms.Resize(32),
    transforms.Lambda(convert_to_rgb),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_ood_normal = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testloader = None
if args.dataset == 'CIFAR10':
    print('CIFAR10')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = datasets.CIFAR10(root='~/data/CIFAR10', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
elif args.dataset == 'CIFAR100':
    print('CIFAR100')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = datasets.CIFAR100(root='~/data/CIFAR100', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
elif args.dataset == 'MNIST':
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.Lambda(convert_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = datasets.MNIST(root='~/data/MNIST', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

'''
Processing OOD data
'''
oodloaders = []
if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
    oodnames = ['SVHN', 'FashionMNIST', 'LSUN', 'TinyImage', 'Corrupted']
    oodloaders.append(loaders.SVHN(transform_ood_normal, batch_size=100, shuffle=False, num_workers=8))
    oodloaders.append(loaders.FashionMNIST(transform_ood_color, batch_size=100, shuffle=False, num_workers=8))
    oodloaders.append(loaders.LSUN_CR(train=False, batch_size=100, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    oodloaders.append(loaders.TinyImageNet(transform_ood_normal, batch_size=100, shuffle=False, num_workers=8))
    oodloaders.append(loaders.Corrupted(args.dataset, transform_noise, batch_size=100, shuffle=False, num_workers=8))
elif args.dataset == 'MNIST':
    oodnames = ['Omniglot', 'FashionMNIST', 'KMNIST', 'CIFAR10', 'Corrupted']
    oodloaders.append(loaders.Omniglot(transform_ood_color, batch_size=100, shuffle=False, num_workers=8))
    oodloaders.append(loaders.FashionMNIST(transform_ood_color, batch_size=100, shuffle=False, num_workers=8))
    oodloaders.append(loaders.KMNIST(transform_ood_color, batch_size=100, shuffle=False, num_workers=8))
    oodloaders.append(loaders.CIFAR10(transform_ood_normal, batch_size=100, shuffle=False, num_workers=8))
    oodloaders.append(loaders.Corrupted(args.dataset, transform_noise, batch_size=100, shuffle=False, num_workers=8))

print('current seeds', args.seed_trail)
if use_cuda:
    torch.manual_seed(args.seed_trail)
    torch.cuda.manual_seed_all(args.seed_trail)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed_trail)
    random.seed(args.seed_trail)
    os.environ['PYTHONHASHSEED'] = str(args.seed_trail)
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint_base = torch.load(
    './checkpoint/ckpt.t7' + args.dataset + '_' + str(args.seed_trail) + '_' + str(args.base_epoch))
checkpoint_meta = torch.load(
    './checkpoint/ckpt.t7' + args.name + '_' + args.meta_model + '_' + str(args.seed_trail))

if use_cuda:
    if args.dataset == 'CIFAR100':
        base_net = models.__dict__[args.base_model](16, 4, 100, 3).cuda()
    else:
        base_net = models.__dict__[args.base_model]().cuda()
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')
    base_net.load_state_dict(checkpoint_base['net'])
    base_net.eval()
    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        meta_net = models.__dict__[args.meta_model](fea_dim1=args.fea_dim[0], fea_dim2=args.fea_dim[1],
                                                    fea_dim3=args.fea_dim[2], fea_dim4=args.fea_dim[3],
                                                    fea_dim5=args.fea_dim[4]).cuda()
    else:
        meta_net = models.__dict__[args.meta_model](fea_dim1=args.fea_dim[0], fea_dim2=args.fea_dim[1]).cuda()
    meta_net.load_state_dict(checkpoint_meta['meta_net'])
    meta_net.eval()


# Collect all uncertainty scores
def get_uncertainty_score(loader, label, get_preds=False):
    assert label in [0, 1]
    label = 1. * label

    base_net.eval()
    meta_net.eval()

    labels = []  # is this point ID (0) or OOD (1)?
    base_ents = []
    base_maxps = []
    base_energy = []

    diff_ents = []
    mis = []
    ents = []
    maxps = []
    precs = []

    base_preds = []
    preds = []

    with torch.no_grad():
        # In distribution data
        for batch_idx, (xs, ys) in enumerate(loader):
            if use_cuda:
                xs, ys = xs.cuda(), ys.cuda()

            base_logits, fea_list = base_net(xs)
            logits = meta_net(*fea_list)

            if get_preds:
                # Get predictions (misclassification binary labels)
                _, base_predicted = torch.max(base_logits.data, 1)
                base_wrongs = base_predicted.ne(ys.data)
                _, meta_predicted = torch.max(logits.data, 1)
                meta_wrongs = meta_predicted.ne(ys.data)

                base_preds.append(base_wrongs.data.cpu())
                preds.append(meta_wrongs.data.cpu())

            # Uncertainty Criterion
            labels.append(label * torch.ones(ys.shape[0]))

            base_ents.append(compute_total_entropy(base_logits).data.cpu())
            base_maxps.append(compute_max_prob(base_logits).data.cpu())
            base_energy.append(torch.logsumexp(base_logits, -1).data.cpu())

            diff_ents.append(compute_differential_entropy(logits).data.cpu())
            mis.append(compute_mutual_information(logits).data.cpu())
            ents.append(compute_total_entropy(logits).data.cpu())
            maxps.append(compute_max_prob(logits).data.cpu())
            precs.append(compute_precision(logits).data.cpu())

        if get_preds:
            base_preds = torch.cat(base_preds, 0)
            preds = torch.cat(preds, 0)

        return torch.cat(diff_ents, 0), \
               torch.cat(mis, 0), \
               torch.cat(ents, 0), \
               torch.cat(maxps, 0), \
               torch.cat(precs, 0), \
               torch.cat(labels, 0), \
               torch.cat(base_ents, 0), torch.cat(base_maxps, 0), torch.cat(base_energy, 0), \
               base_preds, preds


diff_ents, mis, ents, maxps, precs, labels, base_ents, base_maxps, base_energy, base_preds, meta_preds = \
    get_uncertainty_score(testloader, label=0, get_preds=True)

# evaluation
if args.name in ['CIFAR10_miss', 'CIFAR100_miss', 'MNIST_miss']:
    # Evaluate misclassification performance (for test dataset)
    ROC_Selective(diff_ents, mis, ents, maxps, precs,
                  base_ents, base_maxps,
                  base_preds, meta_preds)

elif args.name in ['CIFAR10_OOD', 'CIFAR100_OOD', 'MNIST_OOD']:
    for i in range(len(oodloaders)):
        print(oodnames[i])
        ood_diff_ents, ood_mis, ood_ents, ood_maxps, ood_precs, ood_labels, \
        ood_base_ents, ood_base_maxps, ood_base_energy, *_ = \
            get_uncertainty_score(oodloaders[i], label=1, get_preds=False)

        all_diff_ents = torch.cat([diff_ents, ood_diff_ents])
        all_mis = torch.cat([mis, ood_mis])
        all_ents = torch.cat([ents, ood_ents])
        all_maxps = torch.cat([maxps, ood_maxps])
        all_precs = torch.cat([precs, ood_precs])
        all_labels = torch.cat([labels, ood_labels])
        all_base_ents = torch.cat([base_ents, ood_base_ents])
        all_base_maxps = torch.cat([base_maxps, ood_base_maxps])

        # Evaluate OOD detection performance
        ROC_OOD(all_diff_ents, all_mis, all_ents, all_maxps, all_precs,
                all_labels, all_base_ents, all_base_maxps)
