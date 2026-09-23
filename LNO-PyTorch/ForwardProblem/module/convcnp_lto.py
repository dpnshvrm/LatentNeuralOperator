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

NS2d / y_dim (added 2026-08-27, see claude/convcnp-lno-integration-plan.md's
NS2d section): `y_dim` generalizes the encoder's context-value width from
a hardcoded scalar (1) to any number of channels, so the SAME ConvCNP_LTO
class can encode NS2d's 10-frame sliding-window history (y_dim=10)
instead of just a single steady field (Darcy, y_dim=1, the default --
every existing checkpoint is unaffected). The propagator stays
CNNLatentOperator (no time conditioning) since NS2d's task is a FIXED
single-frame step applied autoregressively (matching LNO's own
train_time/val_time exp.py convention exactly, T_in=10 -> T=10 steps,
sliding window) -- not a continuous/variable dt, which is what
lto/common.py's CNNLatentFlow (used by examples/evolution_heat.py and
the Advection track) is for. This lets the exact same exp.py harness
(train_time/val_time, unmodified) train ConvCNP_LTO on NS2d and produces
a val_loss_full number directly comparable to LNO's own published NS2d
result (paper Table 1, assets/Forward-1.png: 8.45e-2 relative L2) without
retraining LNO itself.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetConvEncoder2D(nn.Module):
    """y_dim (added 2026-08-27, see the NS2d time-evolution section of
    claude/convcnp-lno-integration-plan.md): the encoder's context VALUE
    dimension, generalized from a hardcoded scalar field (y_dim=1) to an
    arbitrary number of channels. This is what lets the same class encode
    a multi-frame history window (e.g. NS2d's 10-frame sliding window,
    y_dim=10) instead of just a single steady field (Darcy, y_dim=1,
    the default -- fully backward compatible, every existing checkpoint
    still loads unchanged since in_channels stays 1+1=2). The first CNN
    layer's in_channels is now 1+y_dim (density channel + y_dim signal
    channels) instead of a hardcoded 2."""

    def __init__(self, grid_size, hidden_channels=16, out_channels=8,
                 init_length_scale=0.1, channel_mode="raw", y_dim=1):
        super().__init__()
        assert channel_mode in ("raw", "normalized"), channel_mode
        self.channel_mode = channel_mode
        g = torch.linspace(0, 1, grid_size)
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        self.register_buffer("grid", torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1))
        self.log_length_scale = nn.Parameter(torch.log(torch.tensor(float(init_length_scale))))
        self.cnn = nn.Sequential(
            nn.Conv2d(1 + y_dim, hidden_channels, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 5, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 5, padding=2),
        )
        self.grid_size = grid_size

    def forward(self, x_ctx, y_ctx):
        # x_ctx: (B, N, 2) in [0,1]^2, y_ctx: (B, N, y_dim)
        ell = torch.exp(self.log_length_scale)
        diff = self.grid.unsqueeze(0).unsqueeze(2) - x_ctx.unsqueeze(1)  # (B, G^2, N, 2)
        w = torch.exp(-0.5 * (diff ** 2).sum(-1) / ell ** 2)  # (B, G^2, N)
        density = w.sum(-1, keepdim=True)
        signal = torch.bmm(w, y_ctx)
        if self.channel_mode == "normalized":
            signal = signal / (density + 1e-8)
        h = torch.cat([density, signal], dim=-1)  # (B, G^2, 1+y_dim)
        B, _, C = h.shape
        h = h.transpose(1, 2).reshape(B, C, self.grid_size, self.grid_size)
        r = self.cnn(h)
        return r.reshape(B, r.shape[1], -1).transpose(1, 2)  # (B, G^2, out_channels)


class AttentiveSetConvEncoder2D(nn.Module):
    """PhCA-style learned cross-attention encoder -- an alternative to
    SetConvEncoder2D's fixed, distance-decaying Gaussian kernel.

    Motivation (see reports/lno_architecture_notes.md's "how are we
    different" section and claude/convcnp-lno-integration-plan.md):
    LNO's PhCA computes its encode weights from a learned score
    (attention_projector(trunk_projector(x))), softmaxed over context
    points, with NO locality/distance bias at all -- a grid location can
    attend strongly to a context point anywhere in the domain if the
    learned score says to. SetConvEncoder2D's Gaussian kernel, by
    contrast, has weight w_i = exp(-||grid_pos - x_i||^2 / (2*ell^2)):
    structurally incapable of putting high weight on a far-away point
    once ell is fixed, no matter what the data says. That locality bias
    is the encoder-side candidate explanation for LNO's edge on Darcy
    (a globally-elliptic PDE) even after the propagator's own receptive
    field was already fixed (use_dilated=True gives the CNN full 64x64
    coverage; only the encoder was still local) and capacity was
    increased (bigcap_resaug/bigcap_resaug_normalized) -- neither
    intervention touches this specific bias, because it lives upstream
    of both.

    Mechanically: query = MLP(grid position), key = MLP(context
    position), value = MLP(context field value); score = scaled dot
    product (standard transformer-style attention, same shape as PhCA's
    own score computation); softmax over context points (dim=-1, same
    normalization axis PhCA uses for its "N points -> tokens" direction).
    Same O(grid_size^2 * n_points) memory/compute cost as
    SetConvEncoder2D's own pairwise kernel -- no new complexity class,
    just a learned weighting function instead of a fixed one.

    THEORY CAVEAT (see claude/convcnp-lno-integration-plan.md and
    reports/lno_architecture_notes.md): Theorem 2.3 / Corollary 2.4's
    discretization-error RATE is derived specifically for a symmetric,
    bandwidth-parameterized kernel (classical kernel-smoothing bias
    theory) and does NOT automatically transfer to this attention
    weighting -- proving an analogous rate bound here is out of scope
    given the paper deadline. Lemma A.1 / Corollary A.2's Lipschitz-
    stability argument only needs non-negative weights summing to 1,
    which the softmax here still guarantees, so that argument plausibly
    DOES still hold, but has not been separately proven for this class.
    Any result trained with use_attention_encoder=True must be reported
    as an empirical architectural ablation, not as evidence for the
    paper's core theorem.

    Drop-in replacement for SetConvEncoder2D: same forward(x_ctx, y_ctx)
    -> (B, grid_size^2, out_channels) signature, so ConvCNP_LTO can swap
    between the two via a constructor flag with no other code changes.
    No channel_mode (raw/normalized) distinction applies here -- the
    softmax-weighted value sum is already a normalized (mean-like)
    combination by construction, same as ConvCNPDecoder2D's own softmax
    weights below.
    """

    def __init__(self, grid_size, x_dim=2, y_dim=1, attn_dim=32, out_channels=8):
        super().__init__()
        g = torch.linspace(0, 1, grid_size)
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        self.register_buffer("grid", torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1))
        self.grid_size = grid_size
        self.attn_dim = attn_dim
        self.query_mlp = nn.Sequential(nn.Linear(x_dim, attn_dim), nn.ReLU(), nn.Linear(attn_dim, attn_dim))
        self.key_mlp = nn.Sequential(nn.Linear(x_dim, attn_dim), nn.ReLU(), nn.Linear(attn_dim, attn_dim))
        self.value_mlp = nn.Sequential(nn.Linear(y_dim, attn_dim), nn.ReLU(), nn.Linear(attn_dim, attn_dim))
        self.cnn = nn.Sequential(
            nn.Conv2d(attn_dim, attn_dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(attn_dim, attn_dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(attn_dim, out_channels, 5, padding=2),
        )

    def forward(self, x_ctx, y_ctx):
        # x_ctx: (B, N, x_dim) in [0,1]^x_dim, y_ctx: (B, N, y_dim)
        B, N, _ = x_ctx.shape
        q = self.query_mlp(self.grid.unsqueeze(0).expand(B, -1, -1))  # (B, G^2, attn_dim)
        k = self.key_mlp(x_ctx)  # (B, N, attn_dim)
        v = self.value_mlp(y_ctx)  # (B, N, attn_dim)
        score = torch.bmm(q, k.transpose(1, 2)) / (self.attn_dim ** 0.5)  # (B, G^2, N)
        w = torch.softmax(score, dim=-1)
        signal = torch.bmm(w, v)  # (B, G^2, attn_dim)
        h = signal.transpose(1, 2).reshape(B, self.attn_dim, self.grid_size, self.grid_size)
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
                 use_dilated=False, use_attention_encoder=False, attn_dim=32, y_dim=1,
                 compute_recon_loss=False):
        super().__init__()
        # y_dim (added 2026-08-27 for the NS2d time-evolution experiment --
        # see claude/convcnp-lno-integration-plan.md): the encoder's
        # context VALUE width. Defaults to 1 (a single steady scalar
        # field, e.g. Darcy's permeability coefficient) -- every existing
        # checkpoint trained so far used this default and is unaffected.
        # NS2d sets y_dim=10 (a 10-frame sliding-window history, matching
        # LNO's own train_time/val_time autoregressive-rollout convention
        # exactly, see configs/LTO_time_NS2d.jsonc) so the SAME class
        # encodes a multi-frame history instead of a single field.
        self.x_dim = x_dim
        self.context_frac_min = context_frac_min
        self.context_frac_max = context_frac_max
        self.use_attention_encoder = use_attention_encoder
        # compute_recon_loss (added 2026-09-23 for the Darcy AE-loss
        # retrofit, see claude/convcnp-lno-integration-plan-part4.md and
        # module/loss.py's RelLpLossWithRecon docstring): default False,
        # so every existing checkpoint/config is unaffected. When True and
        # the model is in training mode, forward() populates
        # self.last_recon_loss (a real, graph-attached tensor) each call;
        # RelLpLossWithRecon reads it back and folds it into the total
        # loss without exp.py's own loss-call structure changing at all.
        self.compute_recon_loss = compute_recon_loss
        self.last_recon_loss = torch.zeros(())  # plain attribute, not a
        # registered buffer -- never enters state_dict, so old checkpoints
        # still load cleanly and this never becomes a persisted weight.
        if use_attention_encoder:
            # PhCA-style learned attention encoder, in place of the fixed
            # Gaussian SetConv kernel -- see AttentiveSetConvEncoder2D's
            # docstring above for the motivation and the theory-scope
            # caveat. channel_mode does not apply here (the softmax
            # weighting is already a normalized combination).
            self.encoder = AttentiveSetConvEncoder2D(grid_size, x_dim=x_dim, y_dim=y_dim,
                                                      attn_dim=attn_dim, out_channels=latent_channels)
        else:
            self.encoder = SetConvEncoder2D(grid_size, hidden_channels, latent_channels,
                                             init_length_scale, channel_mode=channel_mode, y_dim=y_dim)
        self.operator = CNNLatentOperator(grid_size, latent_channels, flow_hidden, use_dilated=use_dilated)
        self.decoder = ConvCNPDecoder2D(grid_size, latent_channels, decoder_hidden, init_length_scale)

    def encode(self, x_ctx, y_ctx):
        return self.encoder(x_ctx, y_ctx)

    def propagate(self, r):
        return self.operator(r)

    def decode(self, x_qry, r):
        return self.decoder(x_qry, r)

    def _densify_context(self, x_full, y_full, resolution):
        """Synthesizes a denser-than-native context via bilinear
        interpolation (align_corners=True) of the real field -- added
        2026-09-23, direct port of train_ns2d_evolution.py's
        make_context_query() extrapolation fix (see that function's
        docstring for the numerical-equivalence argument: bilinear with
        align_corners=True reproduces separable-linear interpolation
        exactly, verified there to ~2e-5 at float32 precision), adapted
        to Darcy's batched (B, N, x_dim) coordinate convention (vs.
        NS2d's shared, unbatched x_coords). Assumes x_full is a flattened
        SQUARE regular grid -- true for every Darcy TRAINING batch (always
        the fixed native 211x211 grid; only the eval script's resolution
        sweep tests other sizes, and this path never runs at eval time)."""
        B, N, x_dim = x_full.shape
        C = y_full.shape[-1]
        gh = gw = int(round(N ** 0.5))
        assert gh * gw == N, "context densification assumes a square native grid"
        x_grid = x_full[0].reshape(gh, gw, x_dim)
        x_axis = x_grid[:, 0, 0]
        y_axis = x_grid[0, :, 1]
        field_grid = y_full.reshape(B, gh, gw, C).permute(0, 3, 1, 2)  # (B, C, gh, gw)
        field_dense = F.interpolate(field_grid, size=(resolution, resolution),
                                     mode="bilinear", align_corners=True)
        y_dense = field_dense.permute(0, 2, 3, 1).reshape(B, resolution * resolution, C)
        new_x = torch.linspace(x_axis.min().item(), x_axis.max().item(), resolution, device=x_full.device)
        new_y = torch.linspace(y_axis.min().item(), y_axis.max().item(), resolution, device=x_full.device)
        xx, yy = torch.meshgrid(new_x, new_y, indexing="ij")
        x_dense_single = torch.stack([xx, yy], dim=-1).reshape(resolution * resolution, x_dim)
        x_dense = x_dense_single.unsqueeze(0).expand(B, -1, -1)
        return x_dense, y_dense

    def _augment_context(self, x_full, y_full):
        """Context-size augmentation, shared by the main forecast context
        and (when compute_recon_loss=True) the recon_t self-encode below
        -- both draw from the SAME sampled frac when called from forward()
        in the same step (consistent with how
        train_burgers_pdebench.py / train_ns2d_evolution.py's recon_t
        term reuses the step's own context_frac) but an INDEPENDENT
        random subset/densification each time this is called.

        frac<=1.0 (original behavior, unchanged): random point subsample.
        frac>1.0 (added 2026-09-23): synthetic denser-than-native grid,
        see _densify_context. context_frac is a POINT-COUNT fraction of
        the native grid (n_ctx / N), NOT a resolution ratio -- e.g.
        Darcy's super-native eval resolution 421 is (421*421)/(211*211) =
        3.98 in this convention, not 421/211 = 2.0 (the same point-count-
        vs-resolution-ratio distinction caught and documented in
        train_ns2d_evolution.py's make_context_query())."""
        B, N, _ = x_full.shape
        frac = torch.empty(1, device=x_full.device).uniform_(self.context_frac_min, self.context_frac_max).item()
        if frac > 1.0:
            gh = int(round(N ** 0.5))
            new_res = max(gh + 1, int(round((frac * N) ** 0.5)))
            x_ctx, y_ctx = self._densify_context(x_full, y_full, new_res)
        else:
            n_ctx = max(1, int(N * frac))
            idx = torch.stack([torch.randperm(N, device=x_full.device)[:n_ctx] for _ in range(B)])
            x_ctx = torch.gather(x_full, 1, idx.unsqueeze(-1).expand(-1, -1, x_full.shape[-1]))
            y_ctx = torch.gather(y_full, 1, idx.unsqueeze(-1).expand(-1, -1, y_full.shape[-1]))
        return x_ctx, y_ctx, frac

    def forward(self, x, y1, x_qry=None, y2=None):
        # y2 (added 2026-09-23, see module/loss.py's RelLpLossWithRecon
        # docstring): optional, only ever passed by exp.py when
        # model_attr["needs_y2"] is True (config.loss.name == "rL2_ae").
        # Every other call site/config passes it as None, unchanged
        # behavior. Not used for the main forecast at all -- only for the
        # recon_t auxiliary term below.
        y_ctx_full = y1[..., self.x_dim:]
        x_ctx, y_ctx = x, y_ctx_full
        # Context-size augmentation, training only (see module docstring) --
        # subsample/densify x/y_ctx to a random fraction of the full dense
        # grid. x_qry defaults to the full, UNAUGMENTED x below, so the
        # forecast loss is always computed against the full dense field
        # regardless.
        if self.training and self.context_frac_min is not None and self.context_frac_max is not None:
            x_ctx, y_ctx, _frac = self._augment_context(x, y_ctx_full)
        r_s = self.encode(x_ctx, y_ctx)
        r_t = self.propagate(r_s)
        x_qry_used = x_qry if x_qry is not None else x
        out = self.decode(x_qry_used, r_t)

        if self.training and self.compute_recon_loss:
            # recon_s: reuse r_s (no re-encode), decode it DIRECTLY -- no
            # propagate() call -- against the FULL dense INPUT
            # (coefficient) field. Classic autoencoder round-trip:
            # decode(encode(y1)) ~= y1. Mirrors
            # train_burgers_pdebench.py / train_ns2d_evolution.py's
            # recon_s_loss exactly.
            recon_s = self.decode(x, r_s)
            recon_s_loss = F.mse_loss(recon_s, y_ctx_full)
            if y2 is not None:
                # recon_t: fresh encode+decode of the TRUE target
                # (solution) field, same augmentation mechanism as the
                # forecast context (independent random draw), no
                # propagate() call. Darcy's direct analog of
                # train_burgers_pdebench.py / train_ns2d_evolution.py's
                # recon_t_loss (history_len==1 case) -- Darcy has no
                # multi-frame trajectory to draw a second real frame
                # from, so y2 (the only other real field the model ever
                # sees) plays that role.
                if self.context_frac_min is not None and self.context_frac_max is not None:
                    x_ctx_t, y_ctx_t, _frac_t = self._augment_context(x, y2)
                else:
                    x_ctx_t, y_ctx_t = x, y2
                r_t_self = self.encode(x_ctx_t, y_ctx_t)
                recon_t = self.decode(x, r_t_self)
                recon_t_loss = F.mse_loss(recon_t, y2)
                self.last_recon_loss = 0.5 * (recon_s_loss + recon_t_loss)
            else:
                self.last_recon_loss = recon_s_loss
        else:
            self.last_recon_loss = torch.zeros((), device=x.device)

        return out
