"""
ConvCNP Latent Twin Operator (LTO), steady-problem variant, ported into
LNO's own training harness (Phase 2 of the ConvCNP/LNO integration plan --
see claude/convcnp-lno-integration-plan.md). Registered here as
`ConvCNP_LTO`, imported into module/model.py, dispatched by model name
"LTO_ConvCNP" in module/utils.py::get_model_data.

This is a hand-kept-in-sync copy of the encoder/decoder/operator classes
from the standalone LTO repo's lto/common.py (SetConvEncoder2D,
ConvCNPDecoder2D, CNNLatentOperator) plus the ConvCNP_LTO wrapper from
examples/darcy_steady.py, adapted to run inside this DDP harness. If the
architecture changes in the LTO repo, mirror the change here too -- there
is currently no shared package import between the two repos.

IMPORTANT (memory): the encoder/decoder do an O(grid_size^2 * n_points)
pairwise computation. `grid_size` is the LATENT grid resolution, which is
independent of the data's own resolution (that decoupling is the whole
point of the architecture / the paper's discretization-invariance claim)
-- do NOT set grid_size to match the full Darcy resolution (211), or the
pairwise tensors blow up to tens of GB per batch (211^2 x 211^2 would be
~63TB). grid_size=64 (see configs/LTO_Darcy.jsonc) keeps the encoder/
decoder pairwise tensors around ~6GB each even though the actual input/
output field is the full dense 211x211 grid -- comfortably within an
H200's memory.

CONTEXT-SIZE AUGMENTATION (added 2026-08-23, see
claude/convcnp-lno-integration-plan.md's resolution-transfer section):
the first trained LTO_Darcy checkpoint (500 epochs, always encoding the
full dense 211x211 grid as context every batch) turned out to generalize
badly to any other spatial resolution -- rL2 improved a lot AT 211
(0.213 -> 0.108) but exploded everywhere else (85: 0.33 -> 18.95). The
same root cause and fix were confirmed on the PDEBench Advection track
(examples/train_advection_pdebench.py's --context_frac_min/--context_frac_max):
the encoder's raw [density, signal] features scale with the total number
of context points, and a model that only ever sees ONE fixed count
overfits its downstream CNN to that one magnitude. `ConvCNP_LTO` now
accepts optional `context_frac_min`/`context_frac_max`: when both are
set (see configs/LTO_Darcy.jsonc), every TRAINING forward pass (i.e.
self.training is True -- exp.py's train() calls model.train(); val()
calls model.eval(), so validation stays dense) subsamples a fresh random
fraction of `x`/`y_ctx` as the encoder's context, while the QUERY side
(`x_qry`, defaulting to the full dense `x`) stays untouched, so the loss
is still computed against the full dense field exactly as before. This
lives entirely inside this LTO-specific class -- exp.py and
module/utils.py's shared training loop are UNCHANGED, so LNO's own
baseline training is completely unaffected (this is deliberate: the
augmentation must never touch code paths LNO's own runs go through, or
the head-to-head comparison stops being fair).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetConvEncoder2D(nn.Module):
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
        diff = x_qry.unsqueeze(2) - self.grid.unsqueeze(0).unsqueeze(0)  # (B, Q, G^2, 2)
        w = torch.exp(-0.5 * (diff ** 2).sum(-1) / ell ** 2)
        w = w / (w.sum(-1, keepdim=True) + 1e-8)
        return self.mlp(torch.bmm(w, r))


class CNNLatentOperator(nn.Module):
    """use_dilated=False (default, matches every checkpoint trained so
    far, including LTO_Darcy_resaug_normalized): the original 3-layer
    plain-conv stack (conv_in/conv_mid/conv_out, all 3x3, dilation 1).
    Receptive field = 7x7 grid cells, REGARDLESS of `hidden` width --
    diagnosed as the likely accuracy bottleneck for globally-supported
    elliptic PDEs like Darcy (see claude/convcnp-lno-integration-plan.md):
    a local CNN cannot represent a solution operator whose dependence is
    global no matter how wide it is, which is exactly why widening alone
    (LTO_Darcy_bigcap_resaug) made accuracy worse, not better.

    use_dilated=True: conv_in (dilation 1) -> 5 dilated conv layers
    (dilations 2,4,8,16,32) -> conv_out (dilation 1, zero-init as always).
    Receptive-field radius = 1+2+4+8+16+32+1 = 64, comfortably covering
    the full 64x64 latent grid (max needed radius is 63) in just 7 conv
    layers total -- an exponentially-growing receptive field (WaveNet/
    dilated-ResNet style) instead of a linear one, buying global reach at
    a modest parameter cost instead of brute-force width scaling.
    Parameter names for the non-dilated path (conv_in/conv_mid/conv_out)
    are unchanged, so use_dilated=False loads any existing checkpoint's
    state_dict without modification."""

    def __init__(self, grid_size, channels=8, hidden=32, use_dilated=False):
        super().__init__()
        self.grid_size = grid_size
        self.use_dilated = use_dilated
        self.conv_in = nn.Conv2d(channels, hidden, 3, padding=1)
        if use_dilated:
            dilations = [2, 4, 8, 16, 32]
            self.dilated_convs = nn.ModuleList([
                nn.Conv2d(hidden, hidden, 3, padding=d, dilation=d) for d in dilations
            ])
        else:
            self.conv_mid = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv_out = nn.Conv2d(hidden, channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, r):
        B, M, C = r.shape
        img = r.transpose(1, 2).reshape(B, C, self.grid_size, self.grid_size)
        h = F.silu(self.conv_in(img))
        if self.use_dilated:
            for conv in self.dilated_convs:
                h = F.silu(conv(h))
        else:
            h = F.silu(self.conv_mid(h))
        delta = self.conv_out(h)
        return (img + delta).reshape(B, C, -1).transpose(1, 2)


class ConvCNP_LTO(nn.Module):
    """forward(x, y1) matches LNO's exp.py calling convention (the model
    name "LTO_ConvCNP" contains no "_single", so model_attr["single"] is
    False and exp.py calls model(x, y1)): takes the dense coordinates `x`
    (dim x_dim) and the LNO dataset's concatenated [x, field] tensor
    `y1`, slices off the leading x_dim columns to recover the bare input
    field, and returns a single predicted tensor at the query points
    (defaults to the same points as x, i.e. the full dense grid) -- no
    dataset.py changes needed."""

    def __init__(self, grid_size, x_dim=2, hidden_channels=16, latent_channels=8,
                 flow_hidden=32, decoder_hidden=64, init_length_scale=0.1,
                 channel_mode="raw", context_frac_min=None, context_frac_max=None,
                 use_dilated=False):
        super().__init__()
        self.x_dim = x_dim
        self.context_frac_min = context_frac_min
        self.context_frac_max = context_frac_max
        self.encoder = SetConvEncoder2D(grid_size, hidden_channels, latent_channels,
                                         init_length_scale, channel_mode=channel_mode)
        self.operator = CNNLatentOperator(grid_size, latent_channels, flow_hidden, use_dilated=use_dilated)
        self.decoder = ConvCNPDecoder2D(grid_size, latent_channels, decoder_hidden, init_length_scale)

    def encode(self, x_ctx, y_ctx):
        return self.encoder(x_ctx, y_ctx)

    def propagate(self, r):
        return self.operator(r)

    def decode(self, x_qry, r):
        return self.decoder(x_qry, r)

    def forward(self, x, y1, x_qry=None):
        y_ctx = y1[..., self.x_dim:]
        x_ctx = x
        # Context-size augmentation, training only (see module docstring) --
        # subsample x/y_ctx to a random fraction of the full dense grid.
        # x_qry defaults to the full, UNSUBSAMPLED x below, so the loss is
        # always computed against the full dense field regardless.
        if self.training and self.context_frac_min is not None and self.context_frac_max is not None:
            B, N, _ = x.shape
            frac = torch.empty(1, device=x.device).uniform_(self.context_frac_min, self.context_frac_max).item()
            n_ctx = max(1, int(N * frac))
            idx = torch.stack([torch.randperm(N, device=x.device)[:n_ctx] for _ in range(B)])
            x_ctx = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            y_ctx = torch.gather(y_ctx, 1, idx.unsqueeze(-1).expand(-1, -1, y_ctx.shape[-1]))
        r = self.encode(x_ctx, y_ctx)
        r = self.propagate(r)
        return self.decode(x_qry if x_qry is not None else x, r)
