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
        self.match_to_labels = config['match_to_labels']


    def calc_ot(self, f_source, f_target, y_source):
        num_source = f_source.shape[0]
        num_target = f_target.shape[0]

        ### Python crashes regularly with POT so switching to GeomLoss
        ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=self.p, blur=self.reg)
        if self.match_to_labels is True:
            cost = ot_loss(y_source, f_target)
        else:
            cost = ot_loss(f_source, f_target)
        return cost


    def adapt(self, config, model, fabric, X_source, y_source, X_target, y_target=[]):
        for i in range(config['num_steps']):
            if config['optimizer'] == 'sam':
                self.opt.zero_grad()
                pred_source = model(X_source)
                pred_target = model(X_target)
                source_loss = self.loss_fun(pred_source, y_source)
                ot_cost = self.calc_ot(pred_source, pred_target, y_source)
                loss = source_loss + self.scale * ot_cost
                fabric.backward(loss)
                def closure():
                    self.opt.zero_grad()
                    pred_source = model(X_source)
                    pred_target = model(X_target)
                    source_loss = self.loss_fun(pred_source, y_source)
                    ot_cost = self.calc_ot(pred_source, pred_target, y_source)
                    loss = source_loss + self.scale * ot_cost
                    fabric.backward(loss)
                    return loss
                self.opt.step(closure)
            elif config['optimizer'] == 'kfac':
                self.opt.zero_grad()
                pred_source = model(X_source)
                pred_target = model(X_target)
                source_loss = self.loss_fun(pred_source, y_source)
                ot_cost = self.calc_ot(pred_source, pred_target, y_source)
                loss = source_loss + self.scale * ot_cost
                fabric.backward(loss)
                config['pre'].step() # this is a bit of a hack for now
                self.opt.step()
            elif config['optimizer'] == 'dfw':
                self.opt.zero_grad()
                pred_source = model(X_source)
                pred_target = model(X_target)
                source_loss = self.loss_fun(pred_source, y_source)
                ot_cost = self.calc_ot(pred_source, pred_target, y_source)
                loss = self.scale * source_loss + ot_cost
                fabric.backward(loss)
                self.opt.step(lambda: float(loss))
            else:
                self.opt.zero_grad()
                pred_source = model(X_source)
                pred_target = model(X_target)
                source_loss = self.loss_fun(pred_source, y_source)
                ot_cost = self.calc_ot(pred_source, pred_target, y_source)
                loss = self.scale * source_loss + ot_cost
                fabric.backward(loss)
                self.opt.step()
            # print(f"WRR loss at step {i}: {loss}")


    def validate(self, config, model, fabric, X_source, y_source, X_target):
        pass


    def checkpoint(self, config, model, fabric, X_source, y_source, X_target, save_path):
        pred_source = model(X_source)
        pred_target = model(X_target)
        source_loss = self.loss_fun(pred_source, y_source)
        ot_cost = self.calc_ot(pred_source, pred_target, y_source)
        val_loss = source_loss + self.scale * ot_cost
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
