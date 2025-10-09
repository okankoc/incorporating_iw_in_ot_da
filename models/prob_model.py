import torch
from torch import nn
import torch.nn.functional as F
import copy


class ProbModel(nn.Module):
    # Take the last penultimate fc layer
    # Add another fc layer with double size, initialized partially from old parameters
    def __init__(self, model):
        super().__init__()
        self.name = "Prob" + model.name
        self.num_classes = model.num_classes
        # Find the penultimate linear layer
        children = list(model.net.named_children())
        for idx, (name, module) in reversed(list(enumerate(children))):
            if idx == len(children) - 1:
                continue
            if "fc" in name:
                old_weight = module.weight
                old_bias = module.bias
                self.net = nn.Sequential(model.net[:idx])
                self.last_layer = model.net[-1]
        old_dim_out = old_weight.shape[0]
        old_dim_in = old_weight.shape[1]
        layer = nn.Linear(old_dim_in, 2 * old_dim_out)
        layer.weight.data[:old_dim_out, :] = old_weight.data
        layer.bias.data[:old_dim_out] = old_bias
        self.net.append(layer)
        self.num_features = 2 * old_dim_out

    def forward(self, input_data):
        features = self.net(input_data)
        mean = features[:, : int(self.num_features / 2)]
        std = features[:, int(self.num_features / 2) :]
        normal_dist = torch.distributions.normal.Normal(mean, F.softplus(std))
        feat_dist = torch.distributions.Independent(
            base_distribution=normal_dist, reinterpreted_batch_ndims=1
        )
        # Reparameterized sample
        sample = feat_dist.rsample()
        output = self.last_layer(sample)
        return output

    def forward_distr(self, input_data):
        features = self.net(input_data)
        mean = features[:, : int(self.num_features / 2)]
        std = features[:, int(self.num_features / 2) :]
        normal_dist = torch.distributions.normal.Normal(mean, F.softplus(std))
        feat_dist = torch.distributions.Independent(
            base_distribution=normal_dist, reinterpreted_batch_ndims=1
        )
        # Reparameterized sample
        sample = feat_dist.rsample()
        output = self.last_layer(sample)
        return mean, std, sample, output, feat_dist

    @torch.no_grad()
    def save_params(self):
        self.state = copy.deepcopy(self.state_dict())

    @torch.no_grad()
    def restore_params(self):
        self.load_state_dict(self.state)
        return dict(self.named_parameters())
