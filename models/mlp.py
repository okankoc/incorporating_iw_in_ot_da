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
        # TODO: These don't need to be stored explicitly now
        self.xs = [None] * len(layer_sizes)
        self.xs_pre = [None] * len(layer_sizes)
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

    # To be used for flattening the Hessian into a matrix
    @torch.no_grad()
    def flatten_hessian(self, hess):
        params = dict(self.named_parameters())
        hess_mat = torch.zeros(self.num_params, self.num_params)
        offset_row = 0
        for name_from, param_from in hess.items():
            offset_col = 0
            for name_to, param in param_from.items():
                num_rows = params[name_from].numel()
                num_cols = params[name_to].numel()
                hess_mat[offset_row : offset_row + num_rows, offset_col : offset_col + num_cols] = param.reshape(
                    num_rows, num_cols
                )
                offset_col += num_cols
            offset_row += num_rows
        return hess_mat

    @torch.no_grad()
    def extract_layer_blocks_from_hessian(self, hess):
        hess_mats = [None] * self.num_layers
        for i in range(self.num_layers):
            w = f"layers.{i}.weight"
            b = f"layers.{i}.bias"
            num_weights = self.layers[i].weight.data.numel()
            num_bias = self.layers[i].bias.data.numel()
            mat11 = hess[w][w].reshape(num_weights, num_weights)
            mat12 = hess[w][b].reshape(num_weights, num_bias)
            mat22 = hess[b][b].reshape(num_bias, num_bias)
            mat = torch.hstack((mat11, mat12))
            mat = torch.vstack((mat, torch.hstack((mat12.T, mat22))))
            hess_mats[i] = mat
        return hess_mats

    # To be used for flattening the Jacobian of the outputs w.r.t. the parameters into a matrix
    @torch.no_grad()
    def flatten_jac(self, jac):
        num_outputs = len(self.layers[-1].bias)
        jac_mat = torch.zeros(num_outputs, self.num_params)
        offset_col = 0
        for name, param in jac.items():
            num_elements = param.numel()
            num_cols = num_elements // num_outputs
            jac_mat[:, offset_col : offset_col + num_cols] = param.reshape(num_outputs, num_cols)
            offset_col += num_cols
        return jac_mat

    # To be used for gradients or weights held outside the class
    @torch.no_grad()
    def flatten(self, inputs):
        views = []
        params = dict(self.named_parameters())
        for name, param in params.items():
            views.append(inputs[name].view(-1))
        return torch.concat(views, 0)

    # To be used to get the parameter dictionary back after the updates
    @torch.no_grad()
    def unflatten(self, p_flat):
        assert len(p_flat) == self.num_params
        offset = 0
        params = dict(self.named_parameters())
        for name, param in params.items():
            numel = param.numel()
            param.copy_(p_flat[offset : offset + numel].unflatten(dim=0, sizes=param.shape))
            offset += numel
        return params

    @torch.no_grad()
    def save_params(self):
        self.state = copy.deepcopy(self.state_dict())

    @torch.no_grad()
    def restore_params(self):
        self.load_state_dict(self.state)
        return dict(self.named_parameters())

    def get_layer(self, layer_id):
        return self.xs[layer_id]

    # Call model after this function to get layer outputs
    def track_features(self, layer_id):
        # Register hooks for the layers you're interested in
        def fun(module, inputs, outputs):
            self.features = outputs

        conv1_hook = self.layers[layer_id].register_forward_hook(fun)

    # In the forward compute, we store also the intermediate pre/post-activation layer features
    def forward(self, x):
        x = self.flatten_input(x)
        self.xs_pre[0] = x
        self.xs[0] = x
        for i in range(self.num_layers):
            self.xs_pre[i + 1] = self.layers[i](self.xs[i])
            if i < self.num_layers - 1:
                self.xs[i + 1] = self.activation(self.xs_pre[i + 1])
            else:
                self.xs[i + 1] = self.xs_pre[i + 1]
        if self.soft_max:
            return nn.Softmax(dim=-1)(self.xs[-1])
        else:
            return self.xs[-1]
