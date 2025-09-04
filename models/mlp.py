"""Multi Layer Perceptron implemented to support custom 2nd order optimization routines."""

import glog as log
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions


# Define model
class MultiLayerPerceptron(nn.Module):
    def __init__(self, layer_sizes, f_nonlinear, soft_max=False):
        super().__init__()
        self.name = "mlp_" + "_".join(map(str, layer_sizes))
        self.layer_sizes = layer_sizes
        self.num_classes = layer_sizes[-1]
        self.num_layers = len(layer_sizes) - 1
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(self.num_layers)])
        self.features = None
        self.activation = f_nonlinear
        self.num_params = 0
        for name, param in self.named_parameters():
            self.num_params += param.numel()
        self.soft_max = soft_max

    def copy(self, device):
        new_model = MultiLayerPerceptron(self.layer_sizes, self.activation).to(device)
        new_model.load_state_dict(self.state_dict())
        new_model.save_params()
        return new_model

    def flatten_input(self, x):
        try:
            return nn.Flatten()(x).type(torch.FloatTensor)
        except (IndexError, TypeError) as err:
            # The input is probably already flat
            return x

    @torch.no_grad()
    def save_params(self):
        self.state = copy.deepcopy(self.state_dict())

    @torch.no_grad()
    def restore_params(self):
        self.load_state_dict(self.state)
        return dict(self.named_parameters())

    # Call model after this function to get layer outputs
    def track_features(self, layer_id):
        # Register hooks for the layers you're interested in
        def fun(module, inputs, outputs):
            self.features = outputs

        conv1_hook = self.layers[layer_id].register_forward_hook(fun)

    # In the forward compute, we store also the intermediate pre/post-activation layer features
    def forward(self, x):
        x = self.flatten_input(x)
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i < self.num_layers - 1:
                x = self.activation(x)
        if self.soft_max:
            return nn.Softmax(dim=-1)(x)
        return x
