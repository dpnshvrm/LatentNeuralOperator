"""
Shared ConvCNP / Latent Twin Operator building blocks for the LNO
integration (see claude/convcnp-lno-integration-plan.md). Both the steady
example (dev_convcnp_darcy.py) and the evolution example
(dev_convcnp_evolution.py) import from here so the two stay in sync and
the eventual port into LNO's module/model.py has one source of truth.

Channel mode (raw vs. normalized)
----------------------------------
SetConvEncoder2D supports two settings of `channel_mode`, corresponding
to the still-open item flagged in claude/project-status.md (Corollary
A.2's remark):

  - "raw" (current default, matches the existing heat notebook and every
    trained checkpoint so far): stacks [density, signal] where
    signal = sum_i w_i * y_i -- the unnormalized SetConv sum. This is
    what's actually been trained on to date.
  - "normalized" (the theory-recommended channel, NOT yet retrained on
    anywhere): stacks [density, signal / (density + eps)] -- the
    Nadaraya-Watson-style locally-weighted MEAN instead of the raw sum.
    Corollary A.2 shows this channel gives a resolution-independent
    encoder Lipschitz constant; the raw channel does not.

Both examples below expose this as a constructor argument specifically
so the raw-vs-normalized ablation (project-status.md's "Next steps" #4)
can be run directly against this codebase instead of only the notebook.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetConvEncoder2D(nn.Module):
    """Gaussian SetConv from scattered/dense context points onto a fixed
    grid_size x grid_size latent grid, then a CNN. grid_size is a
    constructor arg (the notebook used a module-level global)."""

    def __init__(self, grid_size, hidden_channels=16, out_channels=8,
                 init_length_scale=0.1, channel_mode="raw"):
        super().__init__()
        assert channel_mode in ("raw", "normalized"), channel_mode
        self.channel_mode = channel_mode
        g = torch.linspace(0, 1, grid_size)
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        self.register_buffer("grid", torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1))
        self.log_length_scale = nn.Parameter(torch.log(torch.tensor(float(init_length_scale))))
        self.cnn = nn.Sequential(
            nn.Conv2d(2, hidden_channels, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 5, padding=2),
        )
        self.grid_size = grid_size

    def forward(self, x_ctx, y_ctx):
        # x_ctx: (B, N, 2) in [0,1]^2, y_ctx: (B, N, 1)
        ell = torch.exp(self.log_length_scale)
        diff = self.grid.unsqueeze(0).unsqueeze(2) - x_ctx.unsqueeze(1)  # (B, G^2, N, 2)
        w = torch.exp(-0.5 * (diff ** 2).sum(-1) / ell ** 2)  # (B, G^2, N)
        density = w.sum(-1, keepdim=True)
        signal = torch.bmm(w, y_ctx)
        if self.channel_mode == "normalized":
            signal = signal / (density + 1e-8)
        h = torch.cat([density, signal], dim=-1)  # (B, G^2, 2)
        B = h.shape[0]
        h = h.transpose(1, 2).reshape(B, 2, self.grid_size, self.grid_size)
        r = self.cnn(h)
        return r.reshape(B, r.shape[1], -1).transpose(1, 2)  # (B, G^2, out_channels)


class ConvCNPDecoder2D(nn.Module):
    """SetConv attention from query points back onto the latent grid,
    softmax-normalized, then an MLP to scalar output. This one is
    already "normalized" by construction (the decoder's kernel weights
    are softmax-normalized regardless of channel_mode above) -- only the
    ENCODER's raw/normalized choice is what's still open."""

    def __init__(self, grid_size, out_channels=8, hidden_dim=64, init_length_scale=0.1):
        super().__init__()
        g = torch.linspace(0, 1, grid_size)
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        self.register_buffer("grid", torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1))
        self.log_length_scale = nn.Parameter(torch.log(torch.tensor(float(init_length_scale))))
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_qry, r):
        ell = torch.exp(self.log_length_scale)
        diff = x_qry.unsqueeze(2) - self.grid.unsqueeze(0).unsqueeze(0)
        w = torch.exp(-0.5 * (diff ** 2).sum(-1) / ell ** 2)
        w = w / (w.sum(-1, keepdim=True) + 1e-8)
        return self.mlp(torch.bmm(w, r))


class CNNLatentOperator(nn.Module):
    """Steady sibling of CNNLatentFlow below: same residual-CNN
    structure, zero-init final conv (starts as identity), but NO time
    conditioning -- for benchmarks with no time axis (Darcy, Airfoil,
    Elasticity, Plasticity, Pipe)."""

    def __init__(self, grid_size, channels=8, hidden=32):
        super().__init__()
        self.grid_size = grid_size
        self.conv_in = nn.Conv2d(channels, hidden, 3, padding=1)
        self.conv_mid = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv_out = nn.Conv2d(hidden, channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, r):
        B, M, C = r.shape
        img = r.transpose(1, 2).reshape(B, C, self.grid_size, self.grid_size)
        h = F.silu(self.conv_in(img))
        h = F.silu(self.conv_mid(h))
        delta = self.conv_out(h)
        return (img + delta).reshape(B, C, -1).transpose(1, 2)


class CNNLatentFlow(nn.Module):
    """Time-conditioned evolution step, ported from
    code/heat_latent_twin_convcnp.ipynb basically unchanged (grid_size
    was already a constructor arg there). FiLM-conditions a residual CNN
    on a scalar (s, t) time pair; zero-init final conv so it starts as
    identity regardless of s, t."""

    def __init__(self, grid_size, channels=8, hidden=32, time_scale=1.0):
        super().__init__()
        self.grid_size = grid_size
        self.time_scale = time_scale
        self.time_mlp = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.conv_in = nn.Conv2d(channels, hidden, 3, padding=1)
        self.film1 = nn.Linear(hidden, hidden * 2)
        self.conv_mid = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.film2 = nn.Linear(hidden, hidden * 2)
        self.conv_out = nn.Conv2d(hidden, channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, r, s, t):
        B, M, C = r.shape
        img = r.transpose(1, 2).reshape(B, C, self.grid_size, self.grid_size)
        s_n, t_n = s / self.time_scale, t / self.time_scale
        time_emb = self.time_mlp(torch.stack([s_n, t_n], dim=-1))
        h = self.conv_in(img)
        g1, b1 = self.film1(time_emb).chunk(2, dim=-1)
        h = F.silu(h * (1 + g1[..., None, None]) + b1[..., None, None])
        h = self.conv_mid(h)
        g2, b2 = self.film2(time_emb).chunk(2, dim=-1)
        h = F.silu(h * (1 + g2[..., None, None]) + b2[..., None, None])
        delta = self.conv_out(h)
        return (img + delta).reshape(B, C, -1).transpose(1, 2)
