import copy
import torch
from torch import nn


# Define model
class MultiLayerPerceptron(nn.Module):
    def __init__(self, layer_sizes, f_nonlinear):
        super().__init__()
        self.name = "mlp_" + "_".join(map(str, layer_sizes))
        self.layer_sizes = layer_sizes
        self.num_classes = layer_sizes[-1]
        self.num_layers = len(layer_sizes) - 1

        self.net = nn.Sequential()
        self.net.add_module("flatten", nn.Flatten())
        for i in range(self.num_layers):
            self.net.add_module(f"fc{i}", nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < self.num_layers - 1:
                self.net.add_module(f"activation{i}", f_nonlinear)
        self.features = None

    def copy(self, device):
        new_model = MultiLayerPerceptron(self.layer_sizes, self.activation).to(device)
        new_model.load_state_dict(self.state_dict())
        new_model.save_params()
        return new_model

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

        hook = self.net[layer_id].register_forward_hook(fun)

    # In the forward compute, we store also the intermediate pre/post-activation layer features
    def forward(self, x):
        return self.net(x)
