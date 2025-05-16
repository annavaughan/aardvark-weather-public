import torch
import torch.nn as nn

from utils import *


class MLP(nn.Module):
    """
    Multi-layer perceptron
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        h_channels=64,
        h_layers=4,
    ):

        super().__init__()

        def hidden_block(h_channels):
            h = nn.Sequential(
                nn.Linear(h_channels, h_channels),
                nn.ReLU(),
            )
            return h

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, h_channels),
            nn.ReLU(),
            *[hidden_block(h_channels) for _ in range(h_layers)],
            nn.Linear(h_channels, out_channels)
        )

    def forward(self, x):
        return self.mlp(x)
