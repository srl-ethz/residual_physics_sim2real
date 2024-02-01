import torch
import torch.nn as nn

class MLPResidual(torch.nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self, input_size, output_size, hidden_sizes=None, act_fn=torch.nn.ELU()):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [input_size] + [512, 512, 512] + [output_size]
        layers = []
        for i in range(len(hidden_sizes) - 1):
            linear_layer = torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]).double()
            layers.append(linear_layer)
            layers.append(torch.nn.LayerNorm(hidden_sizes[i + 1]).double())
            layers.append(act_fn.double())
        # Remove last activation layer
        layers = layers[:-1]

        self.hidden_sizes = hidden_sizes
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass"""
        x = self.network(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

class ResMLPResidual(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=None, num_mlp_blocks=2, act_fn=torch.nn.ELU()):
        super().__init__()
        mlps = []
        self.act_fn = act_fn
        for i in range(num_mlp_blocks):
            if i == 0:
                if hidden_sizes == None:
                    hidden_sizes = [input_size] + [512, 512, 512] + [output_size]
                else:
                    hidden_sizes = [input_size] + hidden_sizes[1:-1] + [output_size]
                mlps.append(MLPResidual(input_size, output_size, hidden_sizes, act_fn=act_fn))
            else:
                if hidden_sizes is None:
                    hidden_sizes = [output_size] + [512, 512, 512] + [output_size]
                else:
                    hidden_sizes = [output_size] + hidden_sizes[1:-1] + [output_size]
                mlps.append(MLPResidual(output_size, output_size, hidden_sizes, act_fn=act_fn))
        self.hidden_sizes = hidden_sizes
        self.mlps = torch.nn.ModuleList(mlps)
    
    def forward(self, x):
        x = self.mlps[0](x)
        for mlp in self.mlps[1:]:
            x = x + mlp(x)
        return x

    def count_parameters(self):
        num_params = 0
        for mlp in self.mlps:
            num_params += mlp.count_parameters()
        return num_params

class ResMLPResidual2(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, num_block_layer=2, num_mlp_blocks=2, act_fn=torch.nn.ELU()):
        super().__init__()
        mlps = []
        self.act_fn = act_fn
        for i in range(num_mlp_blocks):
            if i == 0:
                hidden_sizes = [input_size] + [hidden_size] * (num_block_layer) + [output_size]
            else:
                hidden_sizes = [output_size] + [hidden_size] * (num_block_layer) + [output_size]
            mlps.append(MLPResidual(input_size, output_size, hidden_sizes, act_fn=act_fn))
        self.hidden_sizes = hidden_sizes
        self.mlps = torch.nn.ModuleList(mlps)
    
    def forward(self, x):
        x = self.mlps[0](x)
        for mlp in self.mlps[1:]:
            x = x + mlp(x)
        return x

    def count_parameters(self):
        num_params = 0
        for mlp in self.mlps:
            num_params += mlp.count_parameters()
        return num_params


class ConvResidual(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=None, num_mlp_blocks=0, act_fn=torch.nn.ELU()):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=3, out_channels=10, kernel_size=5, dtype=torch.double
        )
        self.conv2 = nn.Conv1d(
            in_channels=10, out_channels=40, kernel_size=5, dtype=torch.double
        )
        self.fc1 = nn.Linear(40 * 162, 20 * 162, dtype=torch.double)
        self.fc2 = nn.Linear(20 * 162, 10 * 162, dtype=torch.double)
        self.fc3 = nn.Linear(1620, output_size, dtype=torch.double)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=3)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool1d(x, kernel_size=3)
        x = x.view(-1, 40 * 162)
        x = self.fc1(x)
        x = nn.functional.selu(x)
        x = self.fc2(x)
        x = nn.functional.selu(x)
        x = self.fc3(x)
        return x