# Post-hoc Uncertainty Learning using a Dirichlet Meta-Model
This repository contains official implementation of AAAI 2023 paper [Post-hoc Uncertainty Learning using a Dirichlet Meta-Model](https://arxiv.org/abs/2212.07359).

## Requirements
- python == 3.8.8
- pytorch == 1.10.0
- torchvision == 0.11.1
- numpy, scipy, sklearn, random, argparse, csv, os, time, sys

## Usage
### Training base-model
- LeNet/MNIST
```
python train_base_model.py --model='LeNet_BaseModel' --name='MNIST' --dataset='MNIST' --lr=1e-2 --seed=0 --decay=5e-4 --epoch=20
```
- VGG16/CIFAR10
```
python train_base_model.py --model='VGG16_BaseModel' --name='CIFAR10' --dataset='CIFAR10' --lr=1e-1 --seed=0 --decay=1e-4 --epoch=200
```
- WideResNet-16/CIFAR100
```
python train_base_model.py --model='WideResNet_BaseModel' --name='CIFAR100' --dataset='CIFAR100' --lr=1e-1 --seed=0 --decay=1e-4 --epoch=200
```
### Training meta-model
- For different UQ tasks, simply change the "name", such as --name='CIFAR10_OOD' for OOD detection, and --name='CIFAR10_miss' for Misclassfication.
- MNIST
```
python train_meta_model_combine.py --base_model='LeNet_BaseModel' --base_epoch=20 --meta_model='LeNet_MetaModel_combine' --name='MNIST_OOD' --dataset='MNIST' --lr=1e-1 --seed_trail=0 --decay=1e-4 --epoch=20 --lambda_KL=1e-1 
```
- CIFAR10
```
python train_meta_model_combine.py --base_model='VGG16_BaseModel' --meta_model='VGG16_MetaModel_combine' --name='CIFAR10_OOD' --dataset='CIFAR10' --lr=1e-3 --seed_trail=0 --decay=1e-4 --epoch=20 --lambda_KL=1e-3 
```
- CIFAR100
```
python train_meta_model_combine.py --base_model='WideResNet_BaseModel' --meta_model='WideResNet_MetaModel_combine' --name='CIFAR100_OOD' --dataset='CIFAR100' --lr=1e-2 --seed_trail=0 --decay=1e-4 --epoch=20 --lambda_KL=1e-3 
```
### Evaluate
- MNIST
```
python eval_meta_model.py --base_model='LeNet_BaseModel' --meta_model='LeNet_MetaModel_combine' --name='MNIST_OOD' --dataset='MNIST' --base_epoch=20
```
- CIFAR10
```
python eval_meta_model.py --base_model='VGG16_BaseModel' --meta_model='VGG16_MetaModel_combine' --name='CIFAR10_OOD' --dataset='CIFAR10'
```
- CIFAR100
```
python eval_meta_model.py --base_model='WideResNet_BaseModel' --meta_model='WideResNet_MetaModel_combine' --name='CIFAR100_OOD' --dataset='CIFAR100'
```
### Datasets
- Please manually download LSUN and Tiny ImageNet datasets.
- The dataloader automatically downloads other datasets.

## Reference
This code is based on the following repositories: 
- [Mixup](https://github.com/facebookresearch/mixup-cifar10).
- [lula](https://github.com/wiseodd/lula).



