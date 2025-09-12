import torch
import copy
import geomloss

class ConstrainedWRR:
    def __init__(self, config, fabric, model, loss_fun, opt):
        self.loss_fun = copy.deepcopy(loss_fun)
        self.name = 'WRR'
        fabric.setup(model, opt)
        self.opt = opt
        self.p = config['wrr_norm']
        self.reg = config['wrr_entropy_reg']
        self.thresh = config['wrr_thresh']
        self.scale = config['wrr_scale']
        self.mode = 0

    def calc_ot(self, f_source, f_target):
        num_source = f_source.shape[0]
        num_target = f_target.shape[0]

        ### Python crashes regularly with POT so switching to GeomLoss
        ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=self.p, blur=self.reg)
        return ot_loss(f_source, f_target)


    def adapt(self, config, model, fabric, X_source, y_source, X_target, y_target=[]):

        # STRATEGY 1: SWITCH BETWEEN MINIMIZING OT COST AND SOURCE LOSS
        self.opt.zero_grad()
        pred_source = model(X_source)
        pred_target = model(X_target)
        ot_cost = self.calc_ot(pred_source, pred_target)

        if self.mode == 0:
            fabric.backward(ot_cost)
        else:
            source_loss = self.loss_fun(pred_source, y_source)
            loss = source_loss + self.scale * ot_cost
            fabric.backward(loss)
        self.opt.step()


    def validate(self, config, model, fabric, X_source, y_source, X_target):
        pred_source = model(X_source)
        pred_target = model(X_target)
        ot_cost = self.calc_ot(pred_source, pred_target)

        if ot_cost > self.thresh:
            self.mode = 0 # minimize OT
        else:
            self.mode = 1 # minimize scaled WRR
