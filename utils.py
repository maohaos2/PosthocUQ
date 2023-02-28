import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn import metrics
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.distribution import Distribution


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width = 12

TOTAL_BAR_LENGTH = 36.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def ROC_OOD(ood_Dent, ood_MI, ood_Ent, ood_MaxP, ood_precision, all_label,
            base_Ent, base_MaxP):
    print('OOD Detection!')
    auroc_Dent = metrics.roc_auc_score(all_label.numpy(), ood_Dent.numpy())
    auroc_MI = metrics.roc_auc_score(all_label.numpy(), ood_MI.numpy())
    auroc_Ent = metrics.roc_auc_score(all_label.numpy(), ood_Ent.numpy())
    auroc_MaxP = metrics.roc_auc_score(all_label.numpy(), 1 - ood_MaxP.numpy())
    auroc_precision = metrics.roc_auc_score(all_label.numpy(), -ood_precision.numpy())
    auroc_base_Ent = metrics.roc_auc_score(all_label.numpy(), base_Ent.numpy())
    auroc_base_MaxP = metrics.roc_auc_score(all_label.numpy(), 1 - base_MaxP.numpy())

    print('AUROC score of Differential Entropy is', auroc_Dent)
    print('AUROC score of Mutual Information is', auroc_MI)
    print('AUROC score of Total Entropy is', auroc_Ent)
    print('AUROC score of MaxP is', auroc_MaxP)
    print('AUROC score of precision is', auroc_precision)

    print('AUROC score of Base Model Total Entropy is', auroc_base_Ent)
    print('AUROC score of Base Model MaxP is', auroc_base_MaxP)

    aupr_Dent = metrics.average_precision_score(all_label.numpy(), ood_Dent.numpy())
    aupr_MI = metrics.average_precision_score(all_label.numpy(), ood_MI.numpy())
    aupr_Ent = metrics.average_precision_score(all_label.numpy(), ood_Ent.numpy())
    aupr_MaxP = metrics.average_precision_score(all_label.numpy(), 1 - ood_MaxP.numpy())
    aupr_precision = metrics.average_precision_score(all_label.numpy(), -ood_precision.numpy())
    aupr_base_Ent = metrics.average_precision_score(all_label.numpy(), base_Ent.numpy())
    aupr_base_MaxP = metrics.average_precision_score(all_label.numpy(), 1 - base_MaxP.numpy())

    print('AUPR score of Differential Entropy is', aupr_Dent)
    print('AUPR score of Mutual Information is', aupr_MI)
    print('AUPR score of Total Entropy is', aupr_Ent)
    print('AUPR score of MaxP is', aupr_MaxP)
    print('AUPR score of Precision is', aupr_precision)

    print('AUPR score of Base Model Total Entropy is', aupr_base_Ent)
    print('AUPR score of Base Model MaxP is', aupr_base_MaxP)

    return [auroc_Ent * 100, auroc_MaxP * 100, auroc_MI * 100, auroc_Dent * 100, auroc_precision * 100], \
           [aupr_Ent * 100, aupr_MaxP * 100, aupr_MI * 100, aupr_Dent * 100, aupr_precision * 100], \
           [auroc_base_Ent * 100, auroc_base_MaxP * 100, aupr_base_Ent * 100, aupr_base_MaxP * 100]


def ROC_Selective(ood_Dent, ood_MI, ood_Ent, ood_MaxP, ood_precision,
                  base_Ent, base_MaxP,
                  Base_predicted, Meta_predicted):
    print('Misclssification Detection!')
    Meta_predicted = Meta_predicted.int()
    Base_predicted = Base_predicted.int()
    print(Meta_predicted.sum())
    print(Base_predicted.sum())
    auroc_Dent = metrics.roc_auc_score(Meta_predicted.numpy(), ood_Dent.numpy())
    auroc_MI = metrics.roc_auc_score(Meta_predicted.numpy(), ood_MI.numpy())
    auroc_Ent = metrics.roc_auc_score(Meta_predicted.numpy(), ood_Ent.numpy())
    auroc_MaxP = metrics.roc_auc_score(Meta_predicted.numpy(), 1 - ood_MaxP.numpy())
    auroc_precision = metrics.roc_auc_score(Meta_predicted.numpy(), -ood_precision.numpy())
    auroc_base_Ent = metrics.roc_auc_score(Base_predicted.numpy(), base_Ent.numpy())
    auroc_base_MaxP = metrics.roc_auc_score(Base_predicted.numpy(), 1 - base_MaxP.numpy())
    print('AUROC score of Differential Entropy is', auroc_Dent)
    print('AUROC score of Mutual Information is', auroc_MI)
    print('AUROC score of Total Entropy is', auroc_Ent)
    print('AUROC score of MaxP is', auroc_MaxP)
    print('AUROC score of precision is', auroc_precision)

    print('AUROC score of Base Model Total Entropy is', auroc_base_Ent)
    print('AUROC score of Base Model MaxP is', auroc_base_MaxP)

    aupr_Dent = metrics.average_precision_score(Meta_predicted.numpy(), ood_Dent.numpy())
    aupr_MI = metrics.average_precision_score(Meta_predicted.numpy(), ood_MI.numpy())
    aupr_Ent = metrics.average_precision_score(Meta_predicted.numpy(), ood_Ent.numpy())
    aupr_MaxP = metrics.average_precision_score(Meta_predicted.numpy(), 1 - ood_MaxP.numpy())
    aupr_precision = metrics.average_precision_score(Meta_predicted.numpy(), -ood_precision.numpy())
    aupr_base_Ent = metrics.average_precision_score(Base_predicted.numpy(), base_Ent.numpy())
    aupr_base_MaxP = metrics.average_precision_score(Base_predicted.numpy(), 1 - base_MaxP.numpy())
    print('AUPR score of Differential Entropy is', aupr_Dent)
    print('AUPR score of Mutual Information is', aupr_MI)
    print('AUPR score of Total Entropy is', aupr_Ent)
    print('AUPR score of MaxP is', aupr_MaxP)
    print('AUPR score of Precision is', aupr_precision)

    print('AUPR score of Base Model Total Entropy is', aupr_base_Ent)
    print('AUPR score of Base Model MaxP is', aupr_base_MaxP)

    return [auroc_Ent * 100, auroc_MaxP * 100, aupr_Ent * 100, aupr_MaxP * 100, auroc_base_Ent * 100,
            auroc_base_MaxP * 100, aupr_base_Ent * 100, aupr_base_MaxP * 100]


def convert_to_rgb(x):
    return x.convert("RGB")


# Reference: https://discuss.pytorch.org/t/kernel-density-estimation-as-loss-function/62261/8
class GaussianKDE(Distribution):
    def __init__(self, X, bw):
        """
        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        bw : numeric
          bandwidth for Gaussian kernel
        """
        super(Distribution, self).__init__()
        self.X = X
        self.bw = bw
        self.dims = X.shape[-1]
        self.n = X.shape[0]
        self.mvn = MultivariateNormal(loc=torch.zeros(self.dims),
                                      covariance_matrix=torch.eye(self.dims))

    def sample(self, num_samples):
        idxs = (np.random.uniform(0, 1, num_samples) * self.n).astype(int)
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    def score_samples(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.

        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        if X == None:
            X = self.X
        log_probs = ((-self.dims) * np.log(self.bw) +
                     self.mvn.log_prob((X.unsqueeze(1) - Y) / self.bw)).sum(dim=0) / self.n
        return log_probs

    def log_prob(self, Y):
        """Returns the total log probability of one or more points, `Y`, using
        a Multivariate Normal kernel fit to `X` and scaled using `bw`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated

        Returns
        -------
        log_prob : numeric
          total log probability density for the queried points, `Y`
        """
        X_chunks = self.X.split(1000)
        Y_chunks = Y.split(1000)

        log_prob = 0

        for x in X_chunks:
            for y in Y_chunks:
                log_prob += self.score_samples(y, x).sum(dim=0)

        return log_prob
