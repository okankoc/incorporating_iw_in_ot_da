import torch
from torch import nn
import copy
import torch.nn.utils.spectral_norm as sn

from fDAL import fDALLearner


class FDAL:
    def __init__(self, config, fabric, model, loss_fun, opt):
        self.name = "FDAL"
        print(f"Initializing {self.name}")
        self.opt = opt
        backbone = nn.Sequential(model.net[:-1])
        bottleneck_dim = config['bottleneck_dim']
        taskhead = nn.Sequential(
            sn(nn.Linear(bottleneck_dim, bottleneck_dim)),
            nn.LeakyReLU(),
            nn.Dropout(config['dropout_prob']),
            sn(nn.Linear(bottleneck_dim, model.num_classes)),
        )
        self.clip_grad_val = config['clip_grad_val']
        # taskhead = nn.Sequential(sn(model.net[-1]))
        self.alg = fDALLearner(backbone, taskhead, loss_fun, divergence=config['divergence'], n_classes=model.num_classes).to(fabric.device)
        self.alg.train()

    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        self.opt.zero_grad()
        loss, stats = self.alg((X_source, X_target), y_source)
        # print(f"Task loss: {stats["taskloss"]}, fdal loss: {stats["fdal_loss"]}")
        fabric.backward(loss)
        torch.nn.utils.clip_grad_norm(self.alg.parameters(), self.clip_grad_val)
        self.opt.step()


    def validate(self, model, fabric, X_source, y_source, X_target):
        pass
