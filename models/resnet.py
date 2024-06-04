import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, bottleneck):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(bottleneck, bottleneck),
            nn.GELU(),
            # nn.ReLU()
        )
        # self.fc = nn.Linear(bottleneck, bottleneck)
        self.fc = nn.Identity()

    def forward(self, x):
        return self.fc(x) + self.body(x)


class ResNet(nn.Module):
    name = 'resnet'

    def __init__(self, n_input, n_output, bottleneck=64, n_layers=2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_input, bottleneck),
            nn.GELU(),
            # nn.ReLU(),
            *[ResBlock(bottleneck)] * n_layers,
        )
        self.head = nn.Linear(bottleneck, n_output)

    def forward(self, x):
        return self.head(self.body(x))


class ExtendedResNet(nn.Module):
    name = 'resnet_ext'

    def __init__(self, n_input, n_output, bottleneck=64, n_layers=2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_input, bottleneck),
            nn.GELU(),
            *[ResBlock(bottleneck)] * n_layers,
            nn.Linear(bottleneck, n_output)
        )
        self.final = nn.Sequential(
            nn.Linear(n_output, n_output),
            nn.GELU(),
            nn.Linear(n_output, n_output)
        )

    def forward(self, x):
        return self.body(x), self.final(self.body(x))
