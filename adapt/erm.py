import torch
import copy

class ERM:
    def __init__(self, fabric, model, loss_fun, opt):
        self.mode = mode
        self.name = "ERM"
        print(f"Initializing {self.name}")
        self.loss_fun = copy.deepcopy(loss_fun)
        self.opt = opt

    def adapt(self, config, model, fabric, X_source, y_source, X_target, y_target=[]):
        pred_source = model(X_source)
        loss = self.loss_fun(pred_source, y_source)
        fabric.backward(loss)
        self.opt.step()
        self.opt.zero_grad()
