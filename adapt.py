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
class WWRR:
    def __init__(self, config, fabric, model, loss_fun):
        if config['optimizer'] == 'adam':
            self.opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'sgd':
            self.opt = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
        else:
            raise Exception('Unknown optimizer!')
        fabric.setup(model, self.opt)

        self.loss_fun = copy.deepcopy(loss_fun)
        self.name = 'weighted-WRR'
        self.scale = config['wrr_scale']
        self.p = config['wrr_norm']
        self.reg = config['wrr_entropy_reg']
        self.best_val_loss = torch.inf

    def adapt(self, config, model, fabric, X_source, y_source, X_target):
        self.opt.zero_grad()
        pred_source = model(X_source)
        pred_target = model(X_target)
        source_loss = self.loss_fun(pred_source, y_source)
        # loss matrix
        num_target = pred_target.shape[0]
        w_target = torch.ones(num_target, device=fabric.device) / num_target

        cost_mat = torch.cdist(pred_source, pred_target, 2) ** self.p
        self.loss_fun.reduction = 'none'
        self.losses = self.loss_fun(pred_source, y_source)
        self.loss_fun.reduction = 'mean'
        cost_mat_full = cost_mat + self.losses[:, None]
        self.ot_mat = torch.softmax(-cost_mat_full / self.reg, dim=0) * w_target[None, :]
        self.wrr_weighted_cost = torch.sum(self.ot_mat * cost_mat_full)

        if config['add_source_loss'] is True:
            loss = self.scale * source_loss + self.wrr_weighted_cost
        else:
            loss = self.wrr_weighted_cost

        fabric.backward(loss)
        self.opt.step()


    def checkpoint(self, config, model, fabric, X_source, y_source, X_target, save_path):
        pred_source = model(X_source)
        pred_target = model(X_target)
        source_loss = self.loss_fun(pred_source, y_source)
        # loss matrix
        num_target = pred_target.shape[0]
        w_target = torch.ones(num_target, device=fabric.device) / num_target
        cost_mat = torch.cdist(pred_source, pred_target, 2) ** self.p
        self.loss_fun.reduction = 'none'
        losses = self.loss_fun(pred_source, y_source)
        self.loss_fun.reduction = 'mean'
        cost_mat_full = cost_mat + losses[:, None]
        ot_mat = torch.softmax(-cost_mat_full / self.reg, dim=0) * w_target[None, :]
        wrr_weighted_cost = torch.sum(ot_mat * cost_mat_full)

        val_loss = wrr_weighted_cost
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
        # loss matrix
        num_target = pred_target.shape[0]
        w_target = torch.ones(num_target, device=fabric.device) / num_target
        cost_mat = torch.cdist(pred_source, pred_target, 2) ** self.p
        self.loss_fun.reduction = 'none'
        losses = self.loss_fun(pred_source, y_source)
        self.loss_fun.reduction = 'mean'
        cost_mat_full = cost_mat + losses[:, None]
        ot_mat = torch.softmax(-cost_mat_full / self.reg, dim=0) * w_target[None, :]
        wrr_weighted_cost = torch.sum(ot_mat * cost_mat_full)

        source_weights = torch.sum(ot_mat, dim=1)
        weighted_source_loss = torch.sum(source_weights * losses)
        print(f"Weighted WRR: {wrr_weighted_cost.item()}, weighted_source_loss: {weighted_source_loss.item()}")

        if config['calc_entanglement'] == True:
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
        if config['calc_grad_norms'] == True:
            grad_norms = []
            for w in model.parameters():
                 grad_norms.append(torch.linalg.vector_norm(w.grad, ord=2, dim=None))
            print(f"Grad norms: {grad_norms}")


# Wasserstein Marginal Distance regularized source risk minimization using model outputs
class WRR:
    def __init__(self, config, fabric, model, loss_fun):
        if config['optimizer'] == 'adam':
            self.opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'sgd':
            self.opt = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
        else:
            raise Exception('Unknown optimizer!')
        fabric.setup(model, self.opt)

        self.loss_fun = copy.deepcopy(loss_fun)
        self.name = "WRR"
        self.scale = config['wrr_scale']
        self.p = config['wrr_norm']
        self.reg = config['wrr_entropy_reg']
        self.best_val_loss = torch.inf

    def calc_ot(self, f_source, f_target):
        num_source = f_source.shape[0]
        num_target = f_target.shape[0]

        ### Python crashes regularly with POT so switching to GeomLoss
        ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=self.p, blur=self.reg)
        total_cost = ot_loss(f_source, f_target)
        return total_cost


    def adapt(self, config, model, fabric, X_source, y_source, X_target):
        self.opt.zero_grad()
        pred_source = model(X_source)
        pred_target = model(X_target)
        source_loss = self.loss_fun(pred_source, y_source)
        ot_cost = self.calc_ot(pred_source, pred_target)
        self.loss = source_loss + self.scale * ot_cost
        fabric.backward(self.loss)
        self.opt.step()


    def checkpoint(self, config, model, fabric, X_source, y_source, X_target, save_path):
        pred_source = model(X_source)
        pred_target = model(X_target)
        source_loss = self.loss_fun(pred_source, y_source)
        ot_cost = self.calc_ot(pred_source, pred_target)
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
                 grad_norms.append(torch.linalg.vector_norm(w.grad, ord=2, dim=None))
            print(f"Grad norms: {grad_norms}")
