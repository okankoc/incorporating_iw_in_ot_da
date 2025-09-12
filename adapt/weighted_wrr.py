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
        self.p = config['wrr_norm']
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
            cost_mat = torch.cdist(y_source, pred_target, 2) ** self.p
        else:
            cost_mat = torch.cdist(pred_source, pred_target, 2) ** self.p
            self.loss_fun.reduction = 'none'
            losses = self.loss_fun(pred_source, y_source)
            self.loss_fun.reduction = 'mean'
            cost_mat += cost_mat + losses[:, None]
        ot_mat = torch.softmax(-cost_mat / self.reg, dim=0) * w_target[None, :]
        loss = torch.sum(ot_mat * cost_mat)
        return loss, ot_mat


    def adapt(self, config, model, fabric, X_source, y_source, X_target, y_target=[]):
        self.opt.zero_grad()
        loss, _ = self.calc_loss(model, fabric, X_source, y_source, X_target)
        if self.add_source_loss is True:
            pred_source = model(X_source)
            loss += self.scale * self.loss_fun(pred_source, y_source)
        fabric.backward(loss)
        closure = None
        if config['optimizer'] == 'sam':
            def closure(preds, labels):
                loss = self.calc_loss(model, fabric, X_source, y_source, X_target)
                if self.add_source_loss is True:
                    pred_source = model(X_source)
                    loss += self.scale * self.loss_fun(pred_source, y_source)
                loss.backward()
                return loss
        self.opt.step(closure)


    def checkpoint(self, config, model, fabric, X_source, y_source, X_target, save_path):
        val_loss = self.calc_loss(model, fabric, X_source, y_source, X_target)
        if val_loss < self.best_val_loss:
            print(f"Saving loss {val_loss} as best loss so far")
            # save dictionary with everything needed
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.opt.state_dict(),
                'loss': val_loss
            }
            torch.save(checkpoint, save_path)
            self.best_val_loss = val_loss
        else:
            print(f"Loading previous model with loss {self.best_val_loss} vs. current loss {val_loss}")
            checkpoint = torch.load(save_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
