import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class MLPLayer(nn.Module):
    """
    A multi-layer perceptron (MLP) layer with GELU activation.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden layer features.
        hidden_layers (int): Number of hidden layers.
        out_features (int): Number of output features.

    Raises:
        ValueError: If `hidden_layers` is less than 1 and `hidden_features` is not equal to `out_features`.

    Attributes:
        net (nn.Sequential): The MLP network.

    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()

        if hidden_layers < 1 and hidden_features != out_features:
            raise ValueError('hidden_layers must be at least 1 if hidden_features != out_features')

        self.net = [
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
        ]
        if hidden_layers > 1:
            self.net += [
                            nn.Linear(hidden_features, hidden_features),
                            nn.GELU()] * (hidden_layers - 1)
        self.net += [
            nn.Linear(hidden_features, out_features),
            nn.GELU()]

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """
        Forward pass of the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.net(x)


class Solver(nn.Module):
    """
    Solver module. This module performs the operation :math:`z = \int_{t_0}^{t} \alpha_{\tau} d\tau`
    where :math:`\alpha_{\tau}` refers to the encoded time series.
    Args:
        window_length (int): Length of the window (equal to lookback + horizon).
        latent_features (int): Number of latent features.
        patch_length (int): Length of the patch to integrate each pass.
        MLP_hidden_layers (int): Number of hidden layers in the MLP.
        MLP_hidden_features (int): Number of hidden features in the MLP.
    """

    def __init__(self,
                 window_length,
                 latent_features,
                 patch_length,
                 MLP_hidden_layers,
                 MLP_hidden_features,
                 ):
        super().__init__()

        self.L = window_length
        self.D = latent_features
        self.S = patch_length
        self.hidden_layers = MLP_hidden_layers
        self.hidden_features = MLP_hidden_features

        self.mlp_dudt = MLPLayer(
            in_features=self.D,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            out_features=self.D
        )

        self.mlp_u = MLPLayer(
            in_features=self.D,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            out_features=self.D
        )

        self.mlp_integrate = MLPLayer(
            in_features=self.D,
            hidden_features=self.hidden_features,
            hidden_layers=self.hidden_layers,
            out_features=self.D
        )

    def forward(self, tau):
        """
        Forward pass of the Solver.

        Args:
            tau (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Integrated output tensor.
        """
        batch_size = tau.shape[0]
        dudt = self.mlp_dudt(tau) * -1
        u = self.mlp_u(tau)
        u = u.view(batch_size, -1, self.S, self.D)
        dudt = dudt.view(batch_size, -1, self.S, self.D)
        dudt = torch.flip(dudt, [2])
        dudt = dudt[:, :, :-1, :]
        dudt = torch.cat((u[:, :, -1:, :], dudt), dim=-2)
        intdudt = torch.cumsum(dudt, dim=-2)
        intdudt = torch.flip(intdudt, [2])
        intdudt = intdudt.reshape(batch_size, self.L, self.D)

        intdudt = self.mlp_integrate(intdudt)
        return intdudt