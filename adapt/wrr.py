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


    def adapt(self, config, model, fabric, X_source, y_source, X_target):
        if config['optimizer'] == 'sam':
            self.opt.zero_grad()
            pred_source = model(X_source)
            pred_target = model(X_target)
            source_loss = self.loss_fun(pred_source, y_source)
            ot_cost = self.calc_ot(pred_source, pred_target, y_source)
            self.loss = source_loss + self.scale * ot_cost
            fabric.backward(self.loss)
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
            self.loss = source_loss + self.scale * ot_cost
            fabric.backward(self.loss)
            config['pre'].step() # this is a bit of a hack for now
            self.opt.step()
        else:
            self.opt.zero_grad()
            pred_source = model(X_source)
            pred_target = model(X_target)
            source_loss = self.loss_fun(pred_source, y_source)
            ot_cost = self.calc_ot(pred_source, pred_target, y_source)
            self.loss = source_loss + self.scale * ot_cost
            fabric.backward(self.loss)
            self.opt.step()


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


    def debug(self, config, model, fabric, X_source, y_source, X_target, y_target):
        pred_source = model(X_source)
        pred_target = model(X_target)
        source_loss = self.loss_fun(pred_source, y_source)

        print(f"Unweighted WRR: {self.loss.item()}, source_loss: {source_loss.item()}")
        if config['calc_entanglement'] == True:
            num_source = pred_source.shape[0]
            w_source = torch.ones(num_source, device=fabric.device) / num_source
            num_target = pred_target.shape[0]
            w_target = torch.ones(num_target, device=fabric.device) / num_target
            # The problem with using POT is that Sinkhorn is not converging for low reg.
            cost_mat = (torch.cdist(pred_source, pred_target) ** self.p)
            ot_mat = ot.emd(w_source, w_target, cost_mat, numItermax=5000)
            entanglement = torch.sum(ot_mat * (torch.cdist(y_source, y_target) ** self.p))
            print(f"Entanglement: {entanglement}")
        if config['calc_margin'] == True:
            # Get correct points with matching labels
            # Find the gap between max and second max (by sorting for now)
            pred_sorted_val, pred_sorted_ind = torch.sort(pred_source, dim=1, descending=True)
            correct = (pred_sorted_ind[:, 0] == y_source.argmax(1))
            margin = pred_sorted_val[correct, 0] - pred_sorted_val[correct, 1]
            std_margin, avg_margin = torch.std_mean(margin)
            print(f"Margin mean: {avg_margin}, std: {std_margin}")
        if config['calc_grad_norms'] is True:
            grad_norms = []
            for w in model.parameters():
                 grad_norms.append(torch.linalg.vector_norm(w.grad, ord=2, dim=None).item())
            print(f"Grad norms: {grad_norms}")
