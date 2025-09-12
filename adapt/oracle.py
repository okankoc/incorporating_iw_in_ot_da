import torch
import copy

class Oracle:
    def __init__(self, fabric, model, loss_fun, opt):
        self.name = "LJE"  # low-joint-error
        print(f"Initializing Oracle as {self.name}")
        self.loss_fun = copy.deepcopy(loss_fun)
        self.opt = opt

    def adapt(self, config, model, fabric, X_source, y_source, X_target, y_target=[]):
        pred_source = model(X_source)
        loss = self.loss_fun(pred_source, y_source)
        if len(y_target) > 0:
            loss += self.loss_fun(model(X_target), y_target)
        else:
            print("No target labels provided to LJE oracle!")
        fabric.backward(loss)
        self.opt.step()
        self.opt.zero_grad()
