# Domain adaptation approaches
import torch
from torch import nn
from torch.autograd import Function
import torch.distributions
import torch.nn.functional as F
import numpy as np
import ot
import geomloss
import copy

# Wasserstein Marginal Distance regularized source risk minimization using model outputs
class WRR:
    def __init__(self, fabric, model, loss_fun, learning_rate, weight=False, p=1, scale=1.0, reg=1e-4, debug=False):
        self.loss_fun = copy.deepcopy(loss_fun)
        self.opt = torch.optim.Adam(
            model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        fabric.setup(model, self.opt)
        # self.opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0)
        # For debugging
        self.weight = weight
        if weight is True:
            self.name = 'weighted-WRR'
        else:
            self.name = "WRR"
        self.scale = scale
        self.p = p
        self.reg = reg
        self.debug = debug

    def calc_ot(self, f_source, f_target):
        num_source = f_source.shape[0]
        num_target = f_target.shape[0]

        ### Python crashes regularly with POT so switching to GeomLoss
        ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=self.p, blur=self.reg)
        total_cost = ot_loss(f_source, f_target)

        """
        cost_mat = ot.utils.euclidean_distances(f_source, f_target, squared=self.use_squared_dist)
        scale = torch.max(cost_mat)
        cost_mat = cost_mat / scale
        # Weights of the points
        w_source = torch.ones(num_source) / num_source
        w_target = torch.ones(num_target) / num_target
        prob_mat = ot.emd(a=w_source, b=w_target, M=cost_mat).type(torch.float)
        total_cost = torch.sum(prob_mat * cost_mat)
        """

        return total_cost


    def compute_weighted_wrr(self, device, pred_source, pred_target, y_source):
        # loss matrix
        num_source = pred_source.shape[0]
        num_target = pred_target.shape[0]
        w_target = torch.ones(num_target, device=device) / num_target

        if self.p == 1:
            squared = False
        else:
            squared = True
        cost_mat = self.scale * ot.utils.euclidean_distances(pred_source, pred_target, squared)
        self.loss_fun.reduction = 'none'
        losses = self.loss_fun(pred_source, y_source)
        self.loss_fun.reduction = 'mean'
        cost_mat_full = cost_mat + losses[:, None]
        ot_mat = torch.softmax(-cost_mat_full / self.reg, dim=0) * w_target[None, :]
        wrr_weighted_cost = torch.sum(ot_mat * cost_mat_full)
        if self.debug is True:
            source_loss = torch.mean(losses)
            ot_cost = wrr_weighted_cost - source_loss
            print(f"Weighted WRR: {wrr_weighted_cost.item()}, source_loss: {source_loss.item()}, ot_dist: {ot_cost.item()}")
        return wrr_weighted_cost

    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        pred_source = model(X_source)
        pred_target = model(X_target)
        if self.weight is True:
            loss = self.compute_weighted_wrr(fabric.device, pred_source, pred_target, y_source)
        else:
            ot_cost = self.calc_ot(pred_source, pred_target)
            source_loss = self.loss_fun(pred_source, y_source)
            loss = source_loss + self.scale * ot_cost
            if self.debug is True:
                print(f"Unweighted WRR: {loss.item()}, source_loss: {source_loss.item()}, ot_dist: {ot_cost.item()}")
        fabric.backward(loss)
        # if self.debug is True:
        #     grad_norm = 0
        #     for w in self.model.parameters():
        #         grad_norm += torch.sum(w.grad**2)
        #     grad_norm = torch.sqrt(grad_norm)
        #     print(f"WRR: {loss.item()}, Theta-derivative norm: {grad_norm}")
        self.opt.step()
        self.opt.zero_grad()
