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
import ot      # We don't need POT if we don't need to compute entanglement!

# Weighted Wassertein regularized risk
class WeightedWRR:
    def __init__(self, config, fabric, model, loss_fun, opt):
        self.loss_fun = copy.deepcopy(loss_fun)
        self.name = 'weighted-WRR'
        fabric.setup(model, opt)
        self.opt = opt
        self.scale = config['wrr_scale']
        self.reg = config['wrr_entropy_reg']
        self.best_val_loss = torch.inf
        self.match_to_labels = config['match_to_labels']
        self.add_source_loss = config['add_source_loss']

    def calc_loss(self, model, fabric, X_source, y_source, X_target):
        pred_source = model(X_source)
        pred_target = model(X_target)
        source_loss = self.loss_fun(pred_source, y_source)
        # loss matrix
        num_target = pred_target.shape[0]
        w_target = torch.ones(num_target, device=fabric.device) / num_target

        if self.match_to_labels is True:
            cost_mat = torch.cdist(y_source, pred_target, 2)
        else:
            cost_mat = torch.cdist(pred_source, pred_target, 2)
            source_losses = self.loss_fun(pred_source, y_source, reduction='none')
            cost_mat = cost_mat + source_losses[:, None]
        ot_mat = torch.softmax(-cost_mat / self.reg, dim=0) * w_target[None, :]
        loss = torch.sum(ot_mat * cost_mat)
        w_source = torch.sum(ot_mat, dim=1)
        w_source_loss = torch.sum(w_source * source_losses)
        return loss, ot_mat, w_source_loss


    def adapt(self, config, model, fabric, X_source, y_source, X_target, y_target=[]):
        self.opt.zero_grad()
        loss, ot_mat, w_source_loss = self.calc_loss(model, fabric, X_source, y_source, X_target)
        if self.add_source_loss is True:
            loss = loss + self.scale * self.loss_fun(model(X_source), y_source)
        fabric.backward(loss)
        self.opt.step()
        if config['print_during_opt'] is True:
            w_ot_cost = loss - w_source_loss
            print(f"W-WRR: {loss.item()}, w_ot_cost: {w_ot_cost.item()}, w_source_loss: {w_source_loss.item()}")


    @torch.no_grad()
    def validate(self, config, model, fabric, X_source, y_source, X_target):
        pass
