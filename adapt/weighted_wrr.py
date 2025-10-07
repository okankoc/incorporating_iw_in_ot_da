# Domain adaptation approaches
import time
import torch
from torch import nn
from torch.autograd import Function
import torch.distributions
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
import geomloss
import copy
import ot      # We don't need POT if we don't need to compute entanglement!
from sinkhorn_fot import mm_unbalanced

# Weighted Wassertein regularized risk
class WeightedWRR:
    def __init__(self, config, fabric, model, loss_fun, opt):
        self.loss_fun = copy.deepcopy(loss_fun)
        self.name = 'weighted-WRR'
        fabric.setup(model, opt)
        self.opt = opt
        self.opt2 = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scale = config['wrr_scale']
        self.reg = config['wrr_entropy_reg']
        self.best_val_loss = torch.inf
        self.add_source_loss = config['add_source_loss']

    def calc_loss(self, model, fabric, X_source, y_source, X_target):
        pred_source = model(X_source)
        pred_target = model(X_target)
        source_loss = self.loss_fun(pred_source, y_source)
        # loss matrix
        num_target = pred_target.shape[0]
        w_target = torch.ones(num_target, device=fabric.device) / num_target

        cost_mat = torch.cdist(pred_source, pred_target, 2)
        source_losses = self.loss_fun(pred_source, y_source, reduction='none')
        cost_mat = cost_mat + source_losses[:, None]

        num_source = y_source.shape[0]
        w_source = torch.ones(num_source, device=fabric.device) / num_source
        reg_m = (1.0, 100.0)
        ot_mat = ot.sinkhorn_unbalanced(w_source, w_target, cost_mat, self.reg, reg_m, method='sinkhorn')
        # ot_mat_init = torch.softmax(-cost_mat / self.reg, dim=0) * w_target[None, :]
        # ot_mat_init = torch.clamp(ot_mat_init, min=1e-8)
        # ot_mat = mm_unbalanced(w_source, w_target, cost_mat, reg_m[0], reg_m[1], numItermax=1000, autograd_at_convergence=True, G0=ot_mat_init)
        loss = torch.sum(ot_mat * cost_mat)

        w_source = torch.sum(ot_mat, dim=1)
        w_source_loss = torch.sum(w_source * source_losses)
        return loss, ot_mat, w_source_loss


    def adapt(self, config, model, fabric, X_source, y_source, X_target, y_target=[]):
        self.opt.zero_grad()
        source_loss = self.loss_fun(model(X_source), y_source)
        fabric.backward(source_loss)
        self.opt.step()
        self.opt2.zero_grad()
        loss, ot_mat, w_source_loss = self.calc_loss(model, fabric, X_source, y_source, X_target)
        fabric.backward(loss)
        self.opt2.step()

        # if self.add_source_loss is True:
            # loss = loss + self.scale * self.loss_fun(model(X_source), y_source)
        # fabric.backward(loss)
        # self.opt.step()
        if config['print_during_opt'] is True:
            w_ot_cost = loss - w_source_loss
            print(f"W-WRR: {loss.item()}, w_ot_cost: {w_ot_cost.item()}, w_source_loss: {w_source_loss.item()}")


    @torch.no_grad()
    def validate(self, config, model, fabric, X_source, y_source, X_target):
        pass
