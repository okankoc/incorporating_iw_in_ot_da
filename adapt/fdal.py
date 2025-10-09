import torch
from torch import nn
import copy
import torch.nn.utils.spectral_norm as sn

from fDAL import fDALLearner


class FDAL:
    def __init__(self, config, fabric, model, loss_fun):
        self.name = "FDAL"
        print(f"Initializing {self.name}")
        juncture = config['juncture']
        backbone = model.net[:juncture]
        taskhead = model.net[juncture:]
        self.clip_grad_val = config['clip_grad_val']
        if config['auxhead'] == 'none':
            aux_head = None
        else:
            aux_head = config['auxhead']
        self.learner = fDALLearner(backbone, taskhead, loss_fun, divergence=config['divergence'], n_classes=model.num_classes, aux_head=aux_head, grl_params=config['grl'])

        # FDAL is different from other algorithms in that it keeps its own modules internally, unlike
        # others which take a fixed model from outside!
        self.opt = torch.optim.Adam(
            self.learner.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.learner, self.opt = fabric.setup(self.learner, self.opt)

    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        self.opt.zero_grad()
        loss, stats = self.learner((X_source, X_target), y_source)
        fabric.backward(loss)
        torch.nn.utils.clip_grad_norm(self.learner.parameters(), self.clip_grad_val)
        self.opt.step()
        # print(f"Task loss: {stats["taskloss"]}, fdal loss: {stats["fdal_loss"]}")


    def validate(self, model, fabric, X_source, y_source, X_target):
        pass
