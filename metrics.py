import torch


# Standard entropy loss
def compute_entropy(log_probs):
    return torch.sum(-torch.exp(log_probs) * log_probs, dim=1)


# Entropy for Dirichlet output
def compute_total_entropy(log_alphas):
    log_probs = log_alphas - torch.logsumexp(log_alphas, 1, keepdim=True)
    return compute_entropy(log_probs)


# Max Probability for Dirichlet output
def compute_max_prob(log_alphas):
    log_probs = log_alphas - torch.logsumexp(log_alphas, 1, keepdim=True)
    log_confidence, _ = torch.max(log_probs, 1)
    return torch.exp(log_confidence)


# Differential entropy for Dirichlet output
def compute_differential_entropy(log_alphas):
    alphas = torch.exp(log_alphas)
    alpha0 = torch.exp(torch.logsumexp(log_alphas, 1))
    loss = torch.sum(torch.lgamma(alphas), 1) - torch.lgamma(alpha0) - torch.sum(
        (alphas - 1) * (torch.digamma(alphas) - torch.digamma(alpha0).unsqueeze(-1)), 1)
    return loss


# Mutual Information for Dirichlet output
def compute_mutual_information(log_alphas):
    alphas = torch.exp(log_alphas)
    log_alpha0 = torch.logsumexp(log_alphas, 1)
    alpha0 = torch.exp(log_alpha0)
    log_probs = log_alphas - log_alpha0.unsqueeze(-1)
    loss = -torch.sum(torch.exp(log_probs) * (log_probs -
                                              torch.digamma(alphas + 1) +
                                              torch.digamma(alpha0 + 1).unsqueeze(-1)),
                      1)
    return loss


# Precision for Dirichlet output
def compute_precision(log_alphas):
    log_alpha0 = torch.logsumexp(log_alphas, 1)
    return torch.exp(log_alpha0)


# Data Uncertainty for Dirichlet output
def compute_data_uncertainty(log_alphas):
    log_alpha0 = torch.logsumexp(log_alphas, 1)
    log_probs = log_alphas - log_alpha0.unsqueeze(-1)
    alphas = torch.exp(log_alphas)
    alpha0 = torch.exp(log_alpha0)
    loss = - torch.sum(
        log_probs * (torch.digamma(alphas + 1) -
                     torch.digamma(alpha0 + 1).unsqueeze(-1)),
        1)
    return loss
