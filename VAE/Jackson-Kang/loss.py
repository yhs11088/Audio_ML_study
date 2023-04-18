# source : https://github.com/Jackson-Kang/Pytorch-VAE-tutorial

import torch
import torch.nn.functional as F

def VAEloss(x, x_hat, mean, log_var):
    reconstruction_loss = F.binary_cross_entropy(x_hat, x, reduction = "sum")
    KL_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    loss = reconstruction_loss + KL_divergence
    return loss