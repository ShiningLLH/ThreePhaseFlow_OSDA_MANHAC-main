import torch
import torch.nn.functional as F
import numpy as np

def optimizer_scheduler(optimizer, p, init_lr):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr / (1. + 10 * p) ** 0.75

    return optimizer

def gaussian_kernel(x, y, sigma=1.0):
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    xy = torch.matmul(x, y.t())
    dist = xx + yy.t() - 2 * xy
    return torch.exp(-dist / (2 * sigma ** 2))

def compute_dynamic_threshold(tgt_class_logits, base_t=0.5, delta=0.2, output_size=5):
    # probability distribution
    probs = F.softmax(tgt_class_logits, dim=1)

    # entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    batch_mean_entropy = entropy.mean().item()

    # standardization
    max_entropy = np.log(output_size)
    normalized_entropy = batch_mean_entropy / max_entropy

    # exponential moving average smoothing
    if not hasattr(compute_dynamic_threshold, "smoothed_entropy"):
        compute_dynamic_threshold.smoothed_entropy = normalized_entropy
    else:
        compute_dynamic_threshold.smoothed_entropy = 0.8 * compute_dynamic_threshold.smoothed_entropy + 0.2 * normalized_entropy

    # threshold t
    t = base_t + delta * compute_dynamic_threshold.smoothed_entropy
    return np.clip(t, 0.4, 0.8)