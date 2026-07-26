import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN

class PDEPatchEvolution(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.1):
        super().__init__(); self.d_model = d_model
        self.alpha = nn.Parameter(torch.tensor(-2.0))
        self.beta = nn.Parameter(torch.tensor(-2.0))
        self.tau = nn.Parameter(torch.tensor(-2.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        z_left = torch.cat([z[:, :, :1, :], z[:, :, :-1, :]], dim=2)
        z_right = torch.cat([z[:, :, 1:, :], z[:, :, -1:, :]], dim=2)
        dz = z - z_left
        d2z = z_right - 2.0 * z + z_left
        update = self.alpha * d2z - self.beta * dz
        # return z + self.dropout(self.tau * update)
        self.tau_logit = nn.Parameter(torch.tensor(self.tau))
        tau = torch.sigmoid(self.tau_logit)
        return z + self.dropout(tau * update)

class DifferentialPatchAggregator(nn.Module):
    def __init__(self, seq_len: int = 96, patch_len: int = 16, stride: int = 8, d_model: int = 512, dropout: float = 0.1):
        super().__init__(); self.seq_len = seq_len; self.patch_len = patch_len; self.stride = stride; self.d_model = d_model
        self.patch_embed = nn.Sequential(nn.Dropout(dropout), nn.Linear(patch_len, d_model))
        self.pde_layer = PDEPatchEvolution(d_model=d_model, dropout=dropout)
        # self.pool_query = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
        self.pool_query = torch.nn.Parameter(torch.zeros(1, 1, 1, d_model), requires_grad=True)

    def forward(self, x):
        x_patch = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)      # [B, C, N, P]
        x_left = torch.cat([x_patch[:, :, :1, :], x_patch[:, :, :-1, :]], dim=2)     # [B, C, N, P]
        diff_patch = x_patch - x_left                                               # [B, C, N, P]
        z = self.patch_embed(diff_patch)                                            # [B, C, N, D]
        z = self.pde_layer(z)                                                       # [B, C, N, D]
        score = (z * self.pool_query).sum(dim=-1, keepdim=True)
        weight = torch.softmax(score, dim=2)                                        # [B, C, N, 1]
        new_patch = torch.sum(weight * z, dim=2)                                    # [B, C, D]
        return new_patch

class LinearEncoderMultihead(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 4, feature_num: int = 21, dropout: float = 0.5, bias: bool = True, init_alpha: float = 0.1, init_beta: float = 0.1):
        super().__init__()
        self.embed_dim, self.num_heads, self.feature_num = embed_dim, num_heads, feature_num
        self.head_dim = embed_dim // num_heads

        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.weight_mat = nn.Parameter(torch.zeros(num_heads, feature_num, feature_num), requires_grad=True)
        self.attn_dropout, self.out_dropout = nn.Dropout(dropout), nn.Dropout(dropout)
        init_alpha = min(max(float(init_alpha), 1e-4), 1.0 - 1e-4); init_beta = min(max(float(init_beta), 1e-4), 1.0 - 1e-4)
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(init_alpha / (1.0 - init_alpha)), dtype=torch.float32))
        self.beta_logit = nn.Parameter(torch.tensor(math.log(init_beta / (1.0 - init_beta)), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        if self.v_proj.bias is not None: nn.init.zeros_(self.v_proj.bias)

    def forward(self, coupling: torch.Tensor, x_exp: torch.Tensor):
        B, N, D = x_exp.shape

        c_value = coupling
        alpha, beta = torch.sigmoid(self.alpha_logit), torch.sigmoid(self.beta_logit)
        x_coupled = x_exp * (1-beta)  + beta * c_value
        V = self.v_proj(x_coupled).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # [B,H,N,Dh]
        A = F.normalize(F.softplus(self.weight_mat), p=1, dim=-1)                           # [H,N,N]
        out = torch.matmul(A.unsqueeze(0), V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)                                # [B,N,D]
        out = out + x_exp
        return out                                                                          # [B,N,D]


class DSRL(nn.Module):
    def __init__(self, seq_len: int = 96, patch_len: int = 16, stride: int = 8, d_model: int = 512,
                 dropout: float = 0.1, enc_in: int=21, num_heads:int=4, head_drop=0.):
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.dropout = dropout
        self.enc_in = enc_in
        self.head_drop = head_drop
        self.n_heads = num_heads

        self.model = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.seq_len, self.d_model), nn.GELU(), )
        self.diff = DifferentialPatchAggregator(seq_len=self.seq_len, patch_len=self.patch_len,
                                                stride=self.stride, d_model=self.d_model, dropout=self.dropout)
        self.linear_channel = LinearEncoderMultihead(embed_dim=self.d_model, num_heads=self.n_heads,
                                                        feature_num=self.enc_in, dropout=self.head_drop)

    def forward(self, x):
        x_input = x.permute(0, 2, 1)
        x_patch = self.diff(x_input)
        x_out = self.model(x_input)
        chan_out = self.linear_channel(x_patch, x_out)
        return chan_out

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin
        self.revin_layer = RevIN(self.enc_in, affine=True)
        self.head_drop = configs.head_dropout
        self.n_heads = configs.n_heads

        self.DSRL = DSRL(self.seq_len,self.patch_len,self.stride,self.d_model,
                         self.dropout,self.enc_in,self.n_heads,self.head_drop)

        self.Predictor = nn.Sequential(nn.Dropout(self.dropout),
                                       nn.Linear(self.d_model, self.d_model),nn.GELU(),
                                       nn.Dropout(self.dropout),
                                       nn.Linear(self.d_model, self.pred_len),)

    def forward(self, x, cycle_index):
        # 1
        if self.use_revin:
            x = self.revin_layer(x, mode='norm')
        # 2
        H_out = self.DSRL(x)
        # 2
        y_out = self.Predictor(H_out)

        output = y_out.permute(0, 2, 1)
        if self.use_revin:
            output = self.revin_layer(output, mode='denorm')
        return output
