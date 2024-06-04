import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    name = 'mlp'
    def __init__(self, n_input, n_output, bottleneck=64, n_layers=2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_input, bottleneck),
            nn.ReLU(),
            *[nn.Linear(bottleneck, bottleneck), nn.ReLU()] * n_layers,
        )
        self.head = nn.Linear(bottleneck, n_output)

    def forward(self, x):
        return self.head(self.body(x))


class SmallMLP(nn.Module):
    def __init__(self, input_size):
        super(SmallMLP, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 24)

    def forward(self, x):
        out1 = self.fc1(x)
        out1 = torch.relu(out1)

        out2 = self.fc2(out1)
        out2 = torch.relu(out2)

        out3 = self.fc3(out2)
        return out1, out2, out3
