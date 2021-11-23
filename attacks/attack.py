import torch
from torch.nn import functional as F


class Attack:
    def __init__(self, model, criterion, norm='Linf'):
        self.model = model
        self.criterion = criterion
        self.norm = norm
        self.p = float(self.norm[1:])

    def project(self, perturbation, input, eps):
        if self.norm == 'Linf':
            pert = torch.clamp(perturbation, -eps, eps)
        else:
            pert = F.normalize(perturbation.view(perturbation.shape[0], -1),
                               p=self.p, dim=-1).view(perturbation.shape) * eps
        pert.data.add_(input)
        pert.data.clamp_(0, 1).sub_(input)
        return pert

    def random_initialization(self, perturbation, eps):
        if self.norm == 'Linf':
            return torch.empty_like(perturbation).uniform_(-1, 1) * eps / 4
        else:
            return torch.empty_like(perturbation).normal_(0, eps * eps)

    def normalize_grad(self, grad):
        if self.norm == 'Linf':
            return grad.sign()
        else:
            return F.normalize(grad.view(grad.shape[0], -1), p=self.p, dim=-1).view(grad.shape)

    def perturb(self, x, y=None, targeted=False):
        raise NotImplementedError('You need to define a perturb method!')
