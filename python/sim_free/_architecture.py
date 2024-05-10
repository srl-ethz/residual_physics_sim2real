###
# All forward models have the same interface, where the input is a tensor of shape [batch, seq_len, input_dim] and the output is a tensor of shape [batch, output_dim].
###

import torch
import torch.nn as nn
import math
import numpy as np


# Easy script to pick the desired model
def get_model(model, model_params):
    models = {
        "mlp": MLPModel,
        "resmlp": ResMLP,
    }
    return models.get(model.lower())(**model_params)


### ----- MLP ----- ###
class MLPModel (nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, **kwargs):
        super(MLPModel, self).__init__()
        assert num_layers >= 2, "Number of layers must be greater than or equal to 2"

        # Defining the number of layers and the nodes in each layer
        self.activation = nn.ELU()

        modules = [
            nn.Linear(input_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            self.activation,
        ]
        for _ in range(num_layers-2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.LayerNorm(hidden_dim))
            modules.append(self.activation)
        modules.append(nn.Linear(hidden_dim, output_dim))
        modules.append(nn.LayerNorm(output_dim))

        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        out = self.mlp(x)

        return out


### ----- MLP with Residuals ----- ###
class ResMLP (nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, num_blocks=2, **kwargs):
        super(ResMLP, self).__init__()
        assert num_layers >= 2, "Number of layers must be greater than or equal to 2"
        assert num_blocks >= 2, "Number of blocks must be greater than or equal to 2"

        # Defining the number of layers and the nodes in each layer
        self.activation = nn.ELU()

        modules = [MLPModel(input_dim, hidden_dim, num_layers, output_dim)]
        for _ in range(num_blocks-2):
            modules.append(MLPModel(output_dim, hidden_dim, num_layers, output_dim))
        modules.append(MLPModel(output_dim, hidden_dim, num_layers, output_dim))

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = self.mlp[0](x)
        for mlp in self.mlp[1:]:
            out = out + mlp(out)

        return out


