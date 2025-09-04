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

# Wasserstein Marginal Distance regularized source risk minimization using model outputs
class WRR:
    def __init__(self, config, fabric, model, loss_fun, weight):
        if config['optimizer'] == 'adam':
            self.opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'sgd':
            self.opt = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
        else:
            raise Exception('Unknown optimizer!')
        fabric.setup(model, self.opt)

        self.loss_fun = copy.deepcopy(loss_fun)
        self.weight = weight
        if weight is True:
            self.name = 'weighted-WRR'
        else:
            self.name = "WRR"
        self.scale = config['wrr_scale']
        self.p = config['wrr_norm']
        self.reg = config['wrr_entropy_reg']
        self.debug = config['wrr_debug']

    def calc_ot(self, f_source, f_target):
        num_source = f_source.shape[0]
        num_target = f_target.shape[0]

        ### Python crashes regularly with POT so switching to GeomLoss
        ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=self.p, blur=self.reg)
        total_cost = ot_loss(f_source, f_target)
        return total_cost

    def calc_dist_mat(self, f_source, f_target):
        ''' Replacement for the OT library's ot.utils.euclidean_distances functionality. '''
        n_source = f_source.shape[0]
        n_target = f_target.shape[0]
        f_s_big = f_source.repeat_interleave(n_target, dim=0)
        f_t_big = f_target.repeat(n_source, 1)
        cost_mat = torch.linalg.vector_norm(f_s_big - f_t_big, ord=2, dim=1).reshape(n_source, n_target)
        if self.p == 2:
            return cost_mat ** 2
        return cost_mat


    def compute_weighted_wrr(self, device, pred_source, pred_target, y_source):
        # loss matrix
        num_target = pred_target.shape[0]
        w_target = torch.ones(num_target, device=device) / num_target

        cost_mat = self.scale * self.calc_dist_mat(pred_source, pred_target)
        # cost_mat = self.scale * ot.utils.euclidean_distances(pred_source, pred_target, squared)
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
