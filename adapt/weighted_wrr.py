# Domain adaptation approaches
import time
import copy
import torch
from torch import nn
from torch.autograd import Function
import torch.distributions
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
import geomloss
import copy
import ot
from sinkhorn_uot import mm_unbalanced

# Weighted Wassertein regularized risk
class WeightedWRR:
    def __init__(self, config, fabric, model, loss_fun, opt):
        self.loss_fun = copy.deepcopy(loss_fun)
        self.name = 'weighted-WRR'
        fabric.setup(model, opt)
        self.opt = opt
        self.scale = config['scale']
        self.reg = config['entropy_reg']
        self.reg_m = config['reg_m']
        self.add_source_loss = config['add_source_loss']
        self.uot_alg = config['uot_alg']
        self.uot_init = config['uot_init']
        self.separate_optim = config['separate_optim']
        self.autograd_at_convergence = config['autograd_at_convergence']
        self.print_info = config['print_info']
        self.uot_iter_max = config['uot_iter_max']
        self.opt2 = torch.optim.Adam(model.parameters(), lr=self.opt.defaults['lr'], weight_decay=self.opt.defaults['weight_decay'])


    def calc_loss(self, model, fabric, X_source, y_source, X_target):
        pred_source = model(X_source)
        pred_target = model(X_target)
        num_target = pred_target.shape[0]
        w_target = torch.ones(num_target, device=fabric.device) / num_target
        cost_mat = torch.cdist(pred_source, pred_target, 2)
        source_losses = self.loss_fun(pred_source, y_source, reduction='none')
        cost_mat = cost_mat + source_losses[:, None]
        num_source = y_source.shape[0]
        w_source = torch.ones(num_source, device=fabric.device) / num_source

        if self.uot_alg == 'mm':
            ot_mat_init = None
            if self.uot_init == True:
                ot_mat_init = torch.softmax(-cost_mat / self.reg, dim=0) * w_target[None, :]
                ot_mat_init = torch.clamp(ot_mat_init, min=1e-8)
            ot_mat = mm_unbalanced(w_source, w_target, cost_mat, self.reg_m[0], self.reg_m[1], numItermax=self.uot_iter_max, autograd_at_convergence=self.autograd_at_convergence, G0=ot_mat_init)
        elif self.uot_alg == 'sinkhorn':
            ot_mat = ot.sinkhorn_unbalanced(w_source, w_target, cost_mat, self.reg, self.reg_m, method='sinkhorn')
        else:
            raise Exception('UOT method NOT implemented!')
        loss = torch.sum(ot_mat * cost_mat)

        w_source = torch.sum(ot_mat, dim=1)
        w_source_loss = torch.sum(w_source * source_losses)
        return loss, ot_mat, w_source_loss


    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        if self.separate_optim is True:
            if self.add_source_loss is True:
                self.opt.zero_grad()
                source_loss = self.scale * self.loss_fun(model(X_source), y_source)
                fabric.backward(source_loss)
                self.opt.step()
            self.opt2.zero_grad()
            loss, ot_mat, w_source_loss = self.calc_loss(model, fabric, X_source, y_source, X_target)
            fabric.backward(loss)
            self.opt2.step()
        else:
            self.opt.zero_grad()
            loss, ot_mat, w_source_loss = self.calc_loss(model, fabric, X_source, y_source, X_target)
            if self.add_source_loss is True:
                loss += self.scale * self.loss_fun(model(X_source), y_source)
            fabric.backward(loss)
            self.opt.step()
        if self.print_info is True:
            w_ot_cost = loss - w_source_loss
            print(f"W-WRR: {loss.item()}, w_ot_cost: {w_ot_cost.item()}, w_source_loss: {w_source_loss.item()}")


    @torch.no_grad()
    def validate(self, model, fabric, X_source, y_source, X_target):
        pass
