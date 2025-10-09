# Domain adaptation approaches
import torch
from torch import nn
from torch.autograd import Function
import torch.distributions
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
import geomloss
import copy
import ot  # We don't need POT if we don't need to compute entanglement!


# Wasserstein Marginal Distance regularized source risk minimization using model outputs
class WRR:
    def __init__(self, config, fabric, model, loss_fun, opt):
        self.loss_fun = copy.deepcopy(loss_fun)
        self.name = "WRR"
        self.opt = opt
        self.scale = config["scale"]
        self.p = config["norm"]
        self.reg = config["entropy_reg"]
        self.propagate_labels = config["propagate_labels"]
        self.print_info = config["print_info"]
        model, self.opt = fabric.setup(model, self.opt)

    def calc_ot_loss(self, f_source, f_target):
        ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=self.p, blur=self.reg)
        cost = ot_loss(f_source, f_target)
        return cost

    def calc_ot_map(self, f_source, f_target, device):
        num_source = f_source.shape[0]
        num_target = f_target.shape[0]
        w_source = torch.ones(num_source, device=device) / num_source
        w_target = torch.ones(num_target, device=device) / num_target
        cost_mat = torch.cdist(f_source, f_target, p=2)
        ot_mat = ot.emd(w_source, w_target, cost_mat, numItermax=5000)
        return ot_mat

    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        self.opt.zero_grad()
        pred_source = model(X_source)
        pred_target = model(X_target)
        source_loss = self.loss_fun(pred_source, y_source)
        if self.propagate_labels is True:
            ot_map = self.calc_ot_map(pred_source, pred_target, fabric.device)
            num_target = pred_target.shape[0]
            y_hat_target = (
                num_target * ot_map.T @ torch.argmax(y_source, dim=1).float()
            ).to(torch.int64)
            loss = self.scale * source_loss + self.loss_fun.forward_int(
                pred_target, y_hat_target
            )
        else:
            ot_cost = self.calc_ot_loss(pred_source, pred_target)
            loss = self.scale * source_loss + ot_cost
        fabric.backward(loss)
        self.opt.step()
        if self.print_info is True:
            print(
                f"WRR: {loss.item()}, ot_cost: {ot_cost.item()}, source_loss: {source_loss.item()}"
            )

    def validate(self, config, model, fabric, X_source, y_source, X_target):
        pass
