from torch import roll
import torch.nn as nn
from layers.PDETime_Encoder import Encoder
from layers.PDETime_Solver import Solver
from layers.PDETime_Decoder import Decoder

class PDETime(nn.Module):
    """
    PDETime model for solving long-term multivariate time series forecasting problems.
    Args:
        spatial_dim (int): The number of non-time-index dimensions in the time series.
        temporal_features (int): Number of non-time-index temporal features (e.g. day of week, hour of day).
        temporal_latent_features (int): The number of latent features to be learned from the input features.
        lookback (int): The number of previous time steps to consider for prediction.
        horizon (int): The number of future time steps to predict.
        s_cff (float): Number of frequencies for the Concatenated Fourier Features (CFF) module.
        hidden_features (int): The number of hidden features in the encoder module.
        INR_layers (int): The number of layers in the Implicit Neural Representation (INR) module.
        aggregation_layers (int): The number of aggregation modules in the encoder.
        latent_features (int): The number of latent features used for historical and time-index features.
        patch_length (int): The length of the patches used in the solver module.
        MLP_hidden_layers (int): The number of hidden layers in the MLP module used in the solver.
        MLP_hidden_features (int): The number of hidden features in each layer of the MLP module.
    """

    def __init__(self,
                 dimension,
                 temporal_features,
                 temporal_latent_features,
                 lookback,
                 horizon,
                 s_cff,
                 hidden_features,
                 INR_layers,
                 aggregation_layers,
                 latent_features,
                 patch_length,
                 MLP_hidden_layers,
                 MLP_hidden_features,
                 regularization_strength=0.1
                 ):
        super().__init__()

        self.encoder = Encoder(
            spatial_dim=dimension,
            temporal_features=temporal_features,
            temporal_latent_features=temporal_latent_features,
            lookback=lookback,
            horizon=horizon,
            s_cff=s_cff,
            hidden_features=hidden_features,
            INR_layers=INR_layers,
            aggregation_layers=aggregation_layers,
            latent_features=latent_features
        )

        self.solver = Solver(
            window_length=lookback + horizon,
            latent_features=latent_features,
            patch_length=patch_length,
            MLP_hidden_layers=MLP_hidden_layers,
            MLP_hidden_features=MLP_hidden_features
        )

        self.decoder = Decoder(
            lookback=lookback,
            horizon=horizon,
            lambda_=regularization_strength
        )

    def forward(self, x, t, tau):
        """
        Forward pass of the PDETime model.

        Args:
            x (torch.Tensor): The input tensor representing the historical data.
            t (torch.Tensor): The input tensor representing the temporal features.
            tau (torch.Tensor): The input tensor representing the time index.

        Returns:
            torch.Tensor: The predicted target tensor.
        """
        tau = self.encoder(tau, x, t)
        tau = self.solver(tau)
        target = self.decoder(tau, x)
        return target


class PDETimeLoss(nn.Module):
    """
    Calculates the loss for the PDETime model.

    Args:
        loss_fn (callable): The loss function to be used.
        lookback (int): The number of previous time steps to consider.
        horizon (int): The number of future time steps to predict.

    Returns:
        torch.Tensor: The calculated loss value.
    """

    def __init__(self, loss_fn, lookback, horizon):
        super().__init__()
        self.loss_fn = loss_fn
        self.lookback = lookback
        self.horizon = horizon

    def forward(self, outputs, labels):
        fo_diff = outputs - roll(outputs, 1, dims=-2)
        fo_diff_labels = labels - roll(labels, 1, dims=-2)
        x_tau_0 = outputs[:, self.lookback - 1, :].unsqueeze(-2).repeat(1, self.horizon + self.lookback, 1)
        L_r = self.loss_fn(outputs[:, :-self.horizon, :], labels[:, :-self.horizon, :]) / (self.lookback / self.horizon)
        L_p = self.loss_fn(outputs[:, -self.horizon:, :], labels[:, -self.horizon:, :] - x_tau_0[:, -self.horizon:, :])
        L_f = self.loss_fn(fo_diff[:, -self.horizon:, :], fo_diff_labels[:, -self.horizon:, :])
        return L_r + L_p + L_f