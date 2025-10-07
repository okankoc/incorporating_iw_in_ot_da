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

# Wasserstein Marginal Distance regularized source risk minimization using model outputs
class WRR:
    def __init__(self, config, fabric, model, loss_fun, opt):
        self.loss_fun = copy.deepcopy(loss_fun)
        self.name = 'WRR'
        fabric.setup(model, opt)
        self.opt = opt
        self.scale = config['wrr_scale']
        self.p = config['wrr_norm']
        self.reg = config['wrr_entropy_reg']
        self.best_val_loss = torch.inf
        self.propagate_labels = config['propagate_labels']


    def calc_ot_loss(self, f_source, f_target):
        num_source = f_source.shape[0]
        num_target = f_target.shape[0]

        ### Python crashes regularly with POT so switching to GeomLoss
        ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=self.p, blur=self.reg)
        cost = ot_loss(f_source, f_target)
        # return cost


    def calc_ot_map(self, f_source, f_target, device):
        num_source = f_source.shape[0]
        num_target = f_target.shape[0]
        w_source = torch.ones(num_source, device=device) / num_source
        w_target = torch.ones(num_target, device=device) / num_target
        cost_mat = torch.cdist(f_source, f_target, p=2)
        ot_mat = ot.emd(w_source, w_target, cost_mat, numItermax=5000)
        return ot_mat


    def adapt(self, config, model, fabric, X_source, y_source, X_target, y_target=[]):
        self.opt.zero_grad()
        pred_source = model(X_source)
        pred_target = model(X_target)
        source_loss = self.loss_fun(pred_source, y_source)
        if self.propagate_labels is True:
            ot_map = self.calc_ot_map(pred_source, pred_target, fabric.device)
            num_target = pred_target.shape[0]
            y_hat_target = (num_target * ot_map.T @ torch.argmax(y_source, dim=1).float()).to(torch.int64)
            loss = self.scale * source_loss + self.loss_fun.forward_int(pred_target, y_hat_target)
        else:
            ot_cost = self.calc_ot_loss(pred_source, pred_target)
            loss = self.scale * source_loss + ot_cost
        fabric.backward(loss)
        self.opt.step()
        if config['print_during_opt'] is True:
            print(f"WRR: {loss.item()}, ot_cost: {ot_cost.item()}, source_loss: {source_loss.item()}")
            # calc_w_distance_label_shift(y_source, y_target, model.num_classes)

    def validate(self, config, model, fabric, X_source, y_source, X_target):
        pass



# Assuming Euclidean distance is to be used for W_{1,l} computation,
# the result is equal to \sqrt(2) / 2 * \sum_i |p_i - q_i|
def calc_w_distance_label_shift(y_source, y_target, num_classes):
    p_y = torch.sum(y_source, dim=0)
    p_y /= torch.sum(p_y)
    q_y = torch.sum(y_target, dim=0)
    q_y /= torch.sum(q_y)
    w_1_euclidean_dist = np.sqrt(2) * torch.sum(torch.abs(p_y - q_y)) / 2
    print(f"W1_distance_labels: {w_1_euclidean_dist}")
