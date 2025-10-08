import copy
import torch
from torch import nn
from torch.autograd import Function
import numpy as np

import utils
from models.conv import ConvDomainClassifier
from models.mlp import MultiLayerPerceptron as MLP


# Used to optimize DANN (features are optimized to maximize domain classifier error).
# TODO: Is this necessary? Replace with 2 optimizers: one min. and one max.
# Alternatively, pytorch guide suggests using 'hooks'
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN:
    def __init__(
            self, config, fabric, model, loss_fun, opt):
        layer_to_apply_disc = config['layer_to_apply_disc']
        self.discriminator = config['discriminator']
        self.name = "DANN"
        model.track_features(layer_to_apply_disc)
        self.opt_model = opt
        self.opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.discriminator, self.opt_disc = fabric.setup(self.discriminator, self.opt_disc)
        self.loss_class = copy.deepcopy(loss_fun)
        self.loss_domain = nn.CrossEntropyLoss()
        self.p = 0.0

    def forward_adversarial(self, model, X_data):
        self.p += 0.001
        alpha = 2.0 / (1.0 + np.exp(-10 * self.p)) - 1

        class_output = model(X_data)
        reverse_feature = ReverseLayerF.apply(model.features, alpha)
        domain_output = self.discriminator(reverse_feature)
        return class_output, domain_output


    def adapt(self, model, fabric, X_source, y_source, X_target, y_target=[]):
        self.opt_model.zero_grad()
        self.opt_disc.zero_grad()

        source_batch_size = X_source.shape[0]
        # Feeding in source inputs
        domain_label = utils.one_hot(torch.zeros(source_batch_size, device=fabric.device, dtype=torch.long), 2)
        class_output, domain_output = self.forward_adversarial(model, X_source)
        err_s_label = self.loss_class(class_output, y_source)
        err_s_domain = self.loss_domain(domain_output, domain_label)

        # Feeding in target labels
        target_batch_size = X_target.shape[0]
        domain_label = utils.one_hot(torch.ones(target_batch_size, device=fabric.device, dtype=torch.long), 2)
        _, domain_output = self.forward_adversarial(model, X_target)

        err_t_domain = self.loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label

        fabric.backward(err)
        self.opt_model.step()
        self.opt_disc.step()


    def validate(self, model, fabric, X_train, y_train, X_shift):
        pass
