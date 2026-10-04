import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class SineLayer(nn.Module):
    """
    A custom PyTorch module that applies a sine activation function to the output of a linear layer.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the linear layer will not learn an additive bias. Default is True.
        is_first (bool, optional): If set to True, initializes the weights of the linear layer with a uniform distribution
                                  between -1/in_features and 1/in_features. Default is False.
        omega_0 (float, optional): Frequency parameter for the sine activation function. Default is 30.
    """

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the linear layer.

        This method initializes the weights of the linear layer based on the specified
        initialization strategy. If it is the first layer, the weights are initialized
        uniformly between -1/in_features and 1/in_features. Otherwise, the weights are
        initialized uniformly between -sqrt(6/in_features)/omega_0 and sqrt(6/in_features)/omega_0.

        Note:
            - This method modifies the weights of the linear layer in-place.
            - The initialization strategy depends on the values of `is_first`, `in_features`,
              and `omega_0` attributes.

        Returns:
            None
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        """
        Applies the sine activation function to the output of the linear layer.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features) after applying the sine activation function.
        """
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    """
    Siren (Sinusoidal Representation Network) module.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
        hidden_layers (int): Number of hidden layers.
        out_features (int): Number of output features.
        outermost_linear (bool, optional): Whether the final layer is a linear layer. Defaults to False.
        first_omega_0 (float, optional): Frequency of the first layer. Defaults to 30.
        hidden_omega_0 (float, optional): Frequency of the hidden layers. Defaults to 30.
    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """
        Forward pass of the Siren module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.net(x)


# from https://github.com/jmclong/random-fourier-features-pytorch/
def sample_b(s: float, size: tuple) -> Tensor:
    r"""Matrix of size :attr:`size` sampled from from :math:`\mathcal{N}(0, 2^s)`

    Args:
        s (float): standard deviation
        size (tuple): size of the matrix sampled
    """
    return torch.randn(size) * (2 ** s)


# from https://github.com/jmclong/random-fourier-features-pytorch/
def fourier_encoding(
        v: Tensor,
        b: Tensor) -> Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\cos{2 \pi \mathbf{B} \mathbf{v}} , \sin{2 \pi \mathbf{B} \mathbf{v}})`

    Args:
        v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`
        b (Tensor): projection matrix of shape :math:`(\text{encoded_layer_size}, \text{input_size})`

    Returns:
        Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot \text{encoded_layer_size})`
    """
    vp = 2 * np.pi * v @ b.T
    return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


class CFF(nn.Module):
    """
    Layer for mapping coordinates using random Concatenated Fourier Features.

    Args:
        s (int): number of random Fourier features to create per input feature
        in_features (int): the number of input dimensions
        encoded_size (int): the number of dimensions the `b` matrix maps to

    Raises:
        ValueError:
            If `s`, `in_features`, or `encoded_size` is not provided.

    Attributes:
        s (int): number of random Fourier features to create per input feature

    Methods:
        forward(v: Tensor) -> Tensor:
            Computes the mapping using random Fourier features.

    """

    def __init__(self, s: int = None,
                 in_features: int = None,
                 encoded_size: int = None):
        """
        Initializes a CFF layer.

        Args:
            s (int): number of random Fourier features to create per input feature
            in_features (int): the number of input dimensions
            encoded_size (int): the number of dimensions the `b` matrix maps to

        Raises:
            ValueError:
                If `s`, `in_features`, or `encoded_size` is not provided.
        """
        super().__init__()
        self.s = s

        b_s = [sample_b(s_i, (encoded_size, in_features)) for s_i in range(1, s + 1)]

        for i, b in enumerate(b_s):
            self.register_buffer('b' + str(i + 1), b)

    def forward(self, v: Tensor) -> Tensor:
        """
        Computes the mapping using random Fourier features.

        Args:
            v (Tensor): input tensor of shape `(N, *, input_size)`

        Returns:
            Tensor: Tensor mapping using random fourier features of shape `(N, *, 2 * encoded_size)`
        """
        ff = [fourier_encoding(v, getattr(self, 'b' + str(i + 1))) for i in range(self.s)]
        return torch.cat(ff, dim=-1)


class Tau_INR(nn.Module):
    """
        Implicit Neural Representation with concatenated Fourier features for time index feature

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
        s_cff (int): Scaling factor for CFF.
        num_layers (int): Number of layers in the network.
        out_features (int): Number of output features.

    Raises:
        ValueError: If `hidden_features` is not divisible by 2.

    Attributes:
        net (nn.Sequential): Sequential neural network module.

    Methods:
        forward(tau): Performs forward pass through the network.

    """

    def __init__(self, in_features, hidden_features, s_cff, num_layers, out_features):
        super().__init__()
        if hidden_features % 2 != 0:
            raise ValueError('out_features (d) must be divisible by 2: got %d' % out_features)
        self.net = []
        self.net.append(CFF(s_cff, in_features=in_features, encoded_size=out_features // 2))
        self.net.append(nn.Linear(out_features * s_cff, hidden_features, bias=True))
        self.net.append(nn.GELU())
        if num_layers > 2:
            for i in range(num_layers - 2):
                self.net.append(nn.Linear(hidden_features, hidden_features, bias=True))
                self.net.append(nn.GELU())
        self.net.append(nn.Linear(hidden_features, out_features, bias=True))
        self.net.append(nn.GELU())

        self.net = nn.Sequential(*self.net).float()

    def forward(self, tau):
        """
        Forward pass through the network

        Args:
            tau (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.net(tau)


class CrossAttention(nn.Module):
    """
    CrossAttention module performs cross-attention operation between two input tensors.

    Args:
        input_dim (int): The length of the input tensors.
        kq_dim (int): The length of the keys and queries.
        output_dim (int): The length of the output tensor.
    """

    def __init__(self, input_dim, kq_dim, output_dim):
        super().__init__()
        self.d_out_kq = kq_dim
        self.W_q = nn.Linear(input_dim, kq_dim)
        self.W_k = nn.Linear(input_dim, kq_dim)
        self.W_v = nn.Linear(input_dim, output_dim)

    def forward(self, x_1, x_2):
        """
        Forward pass of the CrossAttention module.

        Args:
            x_1 (torch.Tensor): The first input tensor.
            x_2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The output tensor after cross-attention operation.
        """
        queries_1 = self.W_q(x_1)
        keys_2 = self.W_k(x_2)
        values_2 = self.W_v(x_2)
        attn_scores = queries_1.matmul(keys_2.mT)
        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq ** 0.5, dim=-1
        )

        return attn_weights.matmul(values_2)


class AggLayer(nn.Module):
    """
    AggLayer is a module that performs aggregation operations on input tensors.
    Note attributes with single letter names are the same as the paper.
    TODO: Rename these attributes to more descriptive names.

    Args:
        latent_features (int): The number of latent features.
        temporal_latent_features (int): The number of temporal latent features.
        spatial_dim (int): The spatial dimension.
        lookback (int): The lookback value.
        horizon (int): The horizon value.
        attn_dim (int): The attention dimension.

    Attributes:
        D (int): number of latent features in the representation of tau (time index).
        T (int): number of temporal latent features.
        L (int): lookback length.
        H (int): horizon length.
        C (int): number of channels in time series data.
        attn_dim (int): dimension of the key and query matricies in the attention module.
        tau_linear (nn.Linear): Linear layer for tau.
        t_linear (nn.Linear): Linear layer for t.
        x_linear (nn.Linear): Linear layer for x.
        gelu (nn.GELU): GELU activation function.
        tau_norm_1 (nn.LayerNorm): Layer normalization for tau.
        t_norm_1 (nn.LayerNorm): Layer normalization for t.
        x_norm_1 (nn.LayerNorm): Layer normalization for x.
        attention (CrossAttention): CrossAttention module.
        tau_norm_2 (nn.LayerNorm): Layer normalization for tau.
        t_tau_linear (nn.Linear): Linear layer for concatenating t and tau.
        tau_norm_3 (nn.LayerNorm): Layer normalization for tau.

    """

    def __init__(self,
                 latent_features,
                 temporal_latent_features,
                 spatial_dim,
                 lookback,
                 horizon,
                 attn_dim):
        super().__init__()
        self.D = latent_features
        self.T = temporal_latent_features
        self.L = lookback
        self.H = horizon
        self.C = spatial_dim
        self.attn_dim = attn_dim

        self.tau_linear = nn.Linear(self.D, self.D)
        self.t_linear = nn.Linear(self.T, self.T)
        self.x_linear = nn.Linear(self.D, self.D)

        self.gelu = nn.GELU()
        self.tau_norm_1 = nn.LayerNorm((self.L + self.H, self.D))
        self.t_norm_1 = nn.LayerNorm((self.L + self.H, self.T))
        self.x_norm_1 = nn.LayerNorm((self.C, self.D))

        self.attention = CrossAttention(input_dim=self.D, kq_dim=self.attn_dim, output_dim=self.D)
        self.tau_norm_2 = nn.LayerNorm((self.L + self.H, self.D))
        self.t_tau_linear = nn.Linear(self.T + self.D, self.D)
        self.tau_norm_3 = nn.LayerNorm((self.L + self.H, self.D))

    def forward(self, tau, t, x):
        """
        Forward pass of the AggLayer module.

        Args:
            tau (torch.Tensor): Input tensor for the Implicit Neural Representation (INR) of tau, the time index.
            t (torch.Tensor): Input tensor for the INR of t, the temporal features.
            x (torch.Tensor): Input tensor for the INR of x, the historical data.

        Returns:
            torch.Tensor: Output tensor.

        """
        tau = self.tau_linear(tau)
        t = self.t_linear(t)
        x = self.x_linear(x)
        tau = self.tau_norm_1(self.gelu(tau))
        t = self.t_norm_1(t.sin())
        x = self.x_norm_1(self.gelu(x))
        tau = self.tau_norm_2(tau + self.attention(tau, x))
        t = torch.cat((t, tau), dim=-1)
        tau = self.tau_norm_3(tau + self.t_tau_linear(t))

        return tau


class Encoder(nn.Module):
    """
    Encoder module for PDETime model.

    Args:
        spatial_dim (int): Dimension of the historical features.
        temporal_features (int): Dimension of temporal features.
        temporal_latent_features (int): Number of temporal latent features.
        lookback (int): Number of time steps to look back.
        horizon (int): Number of time steps to predict into the future.
        s_cff (float): Number of random Fourier features to create per timestep.
        hidden_features (int): Number of hidden features in any MLPs.
        INR_layers (int): Number of layers in the INR (Implicit Neural Representation) module.
        aggregation_layers (int): Number of aggregation layers.
        latent_features (int): Number of latent features after encoding.
        outermost_linear (bool): Whether to include an outermost linear layer. Default is False.
        first_omega_0 (int): Frequency parameter for the first layer of Siren. Default is 30.
        hidden_omega_0 (float): Frequency parameter for the hidden layers of Siren. Default is 30.0.
    """

    def __init__(self,
                 spatial_dim,
                 temporal_features,
                 temporal_latent_features,
                 lookback,
                 horizon,
                 s_cff,
                 hidden_features,
                 INR_layers,
                 aggregation_layers,
                 latent_features,
                 outermost_linear=False,
                 first_omega_0=30,
                 hidden_omega_0=30.
                 ):
        super().__init__()

        # letters are the same as the paper
        self.L = lookback
        self.H = horizon
        self.s_cff = s_cff if s_cff is not None else 1
        self.C = spatial_dim
        self.T = temporal_latent_features
        self.D = latent_features
        self.hidden_features = hidden_features
        self.INR_layers = INR_layers
        self.temporal_features = temporal_features

        self.tau_INR = Tau_INR(
            in_features=1,
            hidden_features=self.hidden_features,
            s_cff=self.s_cff,
            num_layers=self.INR_layers,
            out_features=self.D
        )

        self.t_siren = Siren(
            in_features=self.temporal_features,
            hidden_features=self.hidden_features,
            hidden_layers=self.INR_layers,
            out_features=self.T,
            outermost_linear=outermost_linear,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0
        )

        self.x_siren = Siren(
            in_features=self.L,
            hidden_features=self.hidden_features,
            hidden_layers=self.INR_layers,
            out_features=self.D,
            outermost_linear=outermost_linear,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0
        )

        self.agg_layers = [
            AggLayer(
                latent_features=self.D,
                temporal_latent_features=self.T,
                spatial_dim=self.C,
                lookback=self.L,
                horizon=self.H,
                attn_dim=64)
        ]

        self.agg_layers *= aggregation_layers

    def forward(self, tau, x, t):
        """
        Forward pass of the Encoder module.

        Args:
            tau (torch.Tensor): Input tensor for tau, the time index.
            t (torch.Tensor): Input tensor for t, the temporal features.
            x (torch.Tensor): Input tensor for x, the historical data.

        Returns:
            torch.Tensor: Output tensor.
        """
        tau = self.tau_INR(tau.unsqueeze(-1))
        x = self.x_siren(torch.transpose(x, 1, 2))
        t = self.t_siren(t)

        for agg_layer in self.agg_layers:
            tau = agg_layer(tau, t, x)
        return tau