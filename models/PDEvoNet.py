import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


def _to_logit(value: float) -> torch.Tensor:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return torch.tensor(math.log(value / (1.0 - value)), dtype=torch.float32)


class PDEPatchEvolution(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        dropout: float = 0.1,
        init_alpha: float = 0.1,
        init_beta: float = 0.1,
        init_tau: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.alpha_logit = nn.Parameter(_to_logit(init_alpha))
        self.beta_logit = nn.Parameter(_to_logit(init_beta))
        self.tau_logit = nn.Parameter(_to_logit(init_tau))
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_left = torch.cat([z[:, :, :1, :], z[:, :, :-1, :]], dim=2)
        z_right = torch.cat([z[:, :, 1:, :], z[:, :, -1:, :]], dim=2)

        dz = z - z_left
        d2z = z_right - 2.0 * z + z_left

        alpha = torch.sigmoid(self.alpha_logit)
        beta = torch.sigmoid(self.beta_logit)
        tau = torch.sigmoid(self.tau_logit)

        update = alpha * d2z - beta * dz
        return z + self.dropout(tau * update)


class DifferentialPatchAggregator(nn.Module):
    def __init__(
        self,
        seq_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        self.patch_embed = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(patch_len, d_model),
        )

        self.pde_layer = PDEPatchEvolution(
            d_model=d_model,
            dropout=dropout,
        )

        self.pool_query = nn.Parameter(
            torch.zeros(1, 1, 1, d_model),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_patch = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x_left = torch.cat([x_patch[:, :, :1, :], x_patch[:, :, :-1, :]], dim=2)
        diff_patch = x_patch - x_left

        z = self.patch_embed(diff_patch)
        z = self.pde_layer(z)

        score = (z * self.pool_query).sum(dim=-1, keepdim=True)
        weight = torch.softmax(score, dim=2)
        new_patch = torch.sum(weight * z, dim=2)

        return new_patch


class LinearEncoderMultihead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 4,
        feature_num: int = 21,
        dropout: float = 0.0,
        bias: bool = True,
        init_lambda: float = 0.1,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feature_num = feature_num
        self.head_dim = embed_dim // num_heads

        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.weight_mat = nn.Parameter(
            torch.zeros(num_heads, feature_num, feature_num),
            requires_grad=True,
        )

        self.lambda_logit = nn.Parameter(_to_logit(init_lambda))
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)

    def forward(self, coupling: torch.Tensor, x_exp: torch.Tensor) -> torch.Tensor:
        assert coupling.shape == x_exp.shape, (
            f"Expected coupling and x_exp to have the same shape, "
            f"but got coupling={coupling.shape}, x_exp={x_exp.shape}."
        )

        B, N, D = x_exp.shape

        assert N == self.feature_num, (
            f"Expected feature dimension N={self.feature_num}, but got N={N}."
        )
        assert D == self.embed_dim, (
            f"Expected embedding dimension D={self.embed_dim}, but got D={D}."
        )

        lam = torch.sigmoid(self.lambda_logit)
        x_coupled = (1.0 - lam) * x_exp + lam * coupling

        V = self.v_proj(x_coupled)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        A = F.normalize(F.softplus(self.weight_mat), p=1, dim=-1)

        out = torch.matmul(A.unsqueeze(0), V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.dropout(out)

        return out + x_exp


class DSRL(nn.Module):
    def __init__(
        self,
        seq_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 512,
        dropout: float = 0.1,
        enc_in: int = 21,
        num_heads: int = 4,
        head_drop: float = 0.0,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.dropout = dropout
        self.enc_in = enc_in
        self.num_heads = num_heads
        self.head_drop = head_drop

        self.base_encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(seq_len, d_model),
            nn.GELU(),
        )

        self.diff_encoder = DifferentialPatchAggregator(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            dropout=dropout,
        )

        self.channel_encoder = LinearEncoderMultihead(
            embed_dim=d_model,
            num_heads=num_heads,
            feature_num=enc_in,
            dropout=head_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_input = x.permute(0, 2, 1)

        diff_state = self.diff_encoder(x_input)
        base_state = self.base_encoder(x_input)

        out = self.channel_encoder(diff_state, base_state)

        return out


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin
        self.head_drop = configs.head_dropout
        self.num_heads = configs.n_heads

        self.revin_layer = RevIN(self.enc_in, affine=True)

        self.dsrl = DSRL(
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            dropout=self.dropout,
            enc_in=self.enc_in,
            num_heads=self.num_heads,
            head_drop=self.head_drop,
        )

        self.predictor = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len),
        )

    def forward(self, x: torch.Tensor, cycle_index=None) -> torch.Tensor:
        if self.use_revin:
            x = self.revin_layer(x, mode="norm")

        h_out = self.dsrl(x)
        y_out = self.predictor(h_out)

        output = y_out.permute(0, 2, 1)

        if self.use_revin:
            output = self.revin_layer(output, mode="denorm")

        return output
