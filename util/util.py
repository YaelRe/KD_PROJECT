import torch


def atanh(x, eps=1e-6):
    x *= (1. - eps)
    return (torch.log((1 + x) / (1 - x))) * 0.5


def clip(x_i, x, eps):
    x_i.data = torch.max(torch.min(x_i, x + eps), x - eps)
    x_i.data = torch.clamp(x_i, 0, 1)
    return x_i
