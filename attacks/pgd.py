import numpy as np
import torch
from attacks.attack import Attack
from torch import optim


class PGD(Attack):
    def __init__(
            self,
            model,
            criterion,
            norm='Linf',
            n_iter=100,
            n_restarts=1,
            alpha=None,
            rand_init=False):
        super(PGD, self).__init__(model, criterion, norm)

        self.alpha = alpha

        self.n_restarts = n_restarts
        self.n_iter = n_iter

        self.rand_init = rand_init

    def perturb(self, x, y=None, eps=0.001, targeted=False):
        # self.model.requires_grad(False)  # added for memory optimization # Adina Yael!
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        x = x.clone().detach()
        # predict_y = False
        # # if y is None:
        # #     predict_y = True
        # #     y = self.model.predict(x)
        # a_abs = np.abs(eps / self.n_iter) if self.alpha is None else np.abs(self.alpha)
        # multiplier = 1 if targeted else -1
        #
        # best_ps = torch.ones_like(y).to(x)
        # best_pert = torch.zeros_like(x)
        #
        # # self.model.eval()
        # all_succ = torch.zeros(self.n_restarts * (self.n_iter + 1), x.shape[0], dtype=torch.bool).to(x.device)
        # for rest in range(self.n_restarts):
        #     pert = torch.zeros_like(x, requires_grad=True)
        #
        #     if self.rand_init or predict_y:
        #         pert = self.random_initialization(pert, eps)
        #
        #     pert = self.project(pert, x, eps)

            # for k in range(self.n_iter):
            #     curr_iter = rest * (self.n_iter + 1) + k
            #     pert.requires_grad_()
            #     x_i = x + pert
            #     oi = self.model.forward(x_i)
            #     probs = torch.softmax(oi, dim=1)
            #     succ = torch.argmax(oi, dim=1) != y
            #     pi = probs[torch.arange(probs.shape[0]), y].squeeze()
            #     if targeted:
            #         pi = 1. - pi
            #
            #     all_succ[curr_iter] = succ | all_succ[curr_iter - 1]
            #     improve = pi < best_ps
            #     best_pert[improve] = pert[improve]
            #     best_ps[improve] = pi[improve]
            #     best_ps[succ] = 0.
            #
            #     loss = multiplier * self.criterion(oi, y)
            #     # print("loss in inter:" + str(k) + "is: " + str(loss)) ### yael
            #     grad = torch.autograd.grad(loss, [pert])[0].detach()
            #
            #     with torch.no_grad():
            #         grad = self.normalize_grad(grad) * a_abs
            #         pert += grad
            #         pert = self.project(pert, x, eps)

            # with torch.no_grad():
            #     x_i = x + pert
            #     oi = self.model.forward(x_i)
            #     probs = torch.softmax(oi, dim=1)
            #     succ = torch.argmax(oi, dim=1) != y
            #     pi = probs[torch.arange(probs.shape[0]), y].squeeze()
            #     if targeted:
            #         pi = 1. - pi
            #
            #     curr_iter = rest * (self.n_iter + 1) + self.n_iter
            #     all_succ[curr_iter] = succ | all_succ[curr_iter - 1]
            #
            #     improve = pi < best_ps
            #     best_pert[improve] = pert[improve]
            #     best_ps[improve] = pi[improve]
            #     best_ps[succ] = 0.

        # x_a = x + best_pert
        # x_a.detach()
        # o = self.model.forward(x) ### yael
        o = None ### yael
        oi = None ### yael
        x_a = None ### yael
        all_succ = None ### yael
        return x_a, o, oi, all_succ