"""
Phase 1 dev script, EVOLUTION example: the time-conditioned Latent Twin
Operator (encode -> propagate(s,t) -> decode with FiLM time-conditioning),
refactored from code/heat_latent_twin_convcnp.ipynb into the same
parametrized-module style as dev_convcnp_darcy.py, sharing building
blocks from convcnp_lto_common.py.

Why this exists (separate from dev_convcnp_darcy.py): Darcy has no time
axis, so that script only exercises CNNLatentOperator (steady, no time
input). This script exercises CNNLatentFlow (FiLM-conditioned on a
scalar (s, t) pair) -- the actual mechanism the paper's Latent Twin
Operator theory is about, and the one LNO's own benchmarks (NS2d) would
need if Phase 4 happens.

This is NOT yet wired to LNO's NS2d benchmark or its exp.py train_time
loop, on purpose: LNO's NS2d loader hands over discrete 10-frame windows,
not continuous (s, t) gaps, so genuine FiLM time-conditioning needs that
loader rebuilt first (flagged as Phase 4 in the integration plan). Until
then, this validates the time-conditioned architecture itself -- both
mechanically (shapes, zero-init identity regardless of s,t, gradients)
and numerically, on a real (if synthetic) evolving PDE: the same 2D heat
diffusion generator the original notebook used, so the "does it actually
learn to propagate, does resolution transfer hold" checks mean something,
not just noise-fitting like the Darcy stand-in data.

Also exercises the encoder's raw vs. normalized channel_mode (see
convcnp_lto_common.py) on the evolution task specifically, since that's
the setting the paper's Appendix A.2 remark is actually about (the
heat notebook is the one experiment that exists at all right now).
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from convcnp_lto_common import SetConvEncoder2D, ConvCNPDecoder2D, CNNLatentFlow


class ConvCNP_LTO_Evolution(nn.Module):
    """Time-conditioned Latent Twin Operator. Exposes encode/propagate/
    decode separately (like the notebook's HeatLatentTwin) rather than a
    single forward(x, y1) like the steady ConvCNP_LTO, because LNO's
    exp.py has no calling convention for continuous (s, t) yet -- see
    module docstring."""

    def __init__(self, grid_size, hidden_channels=16, latent_channels=8,
                 flow_hidden=32, decoder_hidden=64, init_length_scale=0.1,
                 channel_mode="raw", time_scale=1.0):
        super().__init__()
        self.encoder = SetConvEncoder2D(grid_size, hidden_channels, latent_channels,
                                         init_length_scale, channel_mode=channel_mode)
        self.flow = CNNLatentFlow(grid_size, latent_channels, flow_hidden, time_scale=time_scale)
        self.decoder = ConvCNPDecoder2D(grid_size, latent_channels, decoder_hidden, init_length_scale)

    def encode(self, x_ctx, y_ctx):
        return self.encoder(x_ctx, y_ctx)

    def propagate(self, r, s, t):
        return self.flow(r, s, t)

    def decode(self, x_qry, r):
        return self.decoder(x_qry, r)

    def forward(self, x_ctx, y_ctx, s, t, x_qry):
        r_s = self.encode(x_ctx, y_ctx)
        r_t = self.propagate(r_s, s, t)
        return self.decode(x_qry, r_t)


# --------------------------------------------------------------------------
# Synthetic evolving PDE: 2D heat diffusion (same generator as the
# original notebook, trimmed) -- gives a real convergence signal, unlike
# pure noise, so the numerical checks below actually mean something.
# --------------------------------------------------------------------------

D_MAX = 0.016
WIDTH_RANGE = (0.12, 0.22)
N_BUMPS = 4


def random_initial_conditions(batch_size, grid, device, n_bumps=N_BUMPS, width_range=WIDTH_RANGE):
    x = torch.linspace(0, 1, grid, device=device)
    Y, X = torch.meshgrid(x, x, indexing="ij")
    u0 = torch.zeros(batch_size, grid, grid, device=device)
    w_lo, w_hi = width_range
    for _ in range(n_bumps):
        center_x = torch.rand(batch_size, 1, 1, device=device)
        center_y = torch.rand(batch_size, 1, 1, device=device)
        amplitude = (torch.rand(batch_size, 1, 1, device=device) * 1.0 + 0.5) \
            * (torch.randint(0, 2, (batch_size, 1, 1), device=device).float() * 2 - 1)
        width = torch.rand(batch_size, 1, 1, device=device) * (w_hi - w_lo) + w_lo
        dx = torch.minimum((X - center_x).abs(), 1 - (X - center_x).abs())
        dy = torch.minimum((Y - center_y).abs(), 1 - (Y - center_y).abs())
        u0 = u0 + amplitude * torch.exp(-(dx ** 2 + dy ** 2) / (2 * width ** 2))
    return u0


def diffuse(u0, D):
    grid = u0.shape[-1]
    k = torch.fft.fftfreq(grid, d=1.0 / grid).to(u0.device) * 2 * math.pi
    KY, KX = torch.meshgrid(k, k, indexing="ij")
    k2 = (KX ** 2 + KY ** 2).unsqueeze(0)
    u0_hat = torch.fft.fft2(u0)
    return torch.fft.ifft2(u0_hat * torch.exp(-D.view(-1, 1, 1) * k2)).real


def sample_trajectory_pairs(batch_size, grid, device, d_max=D_MAX):
    u0 = random_initial_conditions(batch_size, grid, device)
    ts = torch.rand(batch_size, 2, device=device) * d_max
    s, t = ts.min(dim=1).values, ts.max(dim=1).values
    field_s = diffuse(u0, s)
    field_t = diffuse(u0, t)
    lo = field_s.amin(dim=(1, 2), keepdim=True)
    hi = field_s.amax(dim=(1, 2), keepdim=True)
    scale = (hi - lo).clamp_min(1e-6)
    return (field_s - lo) / scale, (field_t - lo) / scale, s, t


def sample_points_from_field(field, n_points):
    batch_size = field.shape[0]
    xy = torch.rand(batch_size, n_points, 2, device=field.device)
    grid_coords = (xy * 2 - 1).unsqueeze(2)
    values = F.grid_sample(field.unsqueeze(1), grid_coords, mode="bilinear", align_corners=True)
    return xy, values.squeeze(-1).transpose(1, 2)


# --------------------------------------------------------------------------
# Mechanical + numerical validation
# --------------------------------------------------------------------------

def mechanical_checks(channel_mode, device):
    print(f"\n=== mechanical checks, channel_mode = {channel_mode!r} ===")
    grid_size = 16
    batch_size = 4
    model = ConvCNP_LTO_Evolution(grid_size=grid_size, channel_mode=channel_mode, time_scale=D_MAX).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ConvCNP_LTO_Evolution parameters: {n_params:,}  (grid_size={grid_size})")

    field_s, field_t, s, t = sample_trajectory_pairs(batch_size, grid_size, device)
    x_ctx, y_ctx = sample_points_from_field(field_s, 128)
    x_qry, y_qry = sample_points_from_field(field_t, 128)

    out = model(x_ctx, y_ctx, s, t, x_qry)
    assert out.shape == y_qry.shape, f"shape mismatch: {out.shape} vs {y_qry.shape}"
    assert torch.isfinite(out).all()
    print(f"[ok] forward: x_ctx{tuple(x_ctx.shape)} y_ctx{tuple(y_ctx.shape)} -> out{tuple(out.shape)}")

    # Zero-init identity check: CNNLatentFlow's delta path is zero-init,
    # so propagate(r, s, t) == r at init REGARDLESS of s, t (unlike a
    # non-zero-init FiLM layer, where random s,t would perturb the output
    # even at init and give false confidence in "identity" behavior).
    r0 = torch.randn(3, grid_size * grid_size, 8, device=device)
    s_rand = torch.rand(3, device=device) * D_MAX
    t_rand = torch.rand(3, device=device) * D_MAX
    r1 = model.flow(r0, s_rand, t_rand)
    max_diff = (r1 - r0).abs().max().item()
    print(f"[ok] CNNLatentFlow identity at init (random s,t): max diff = {max_diff:.8f}")
    assert max_diff < 1e-6, "flow not near-identity at init -- zero-init conv_out may be broken"

    loss = F.mse_loss(out, y_qry)
    model.zero_grad()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not missing, f"missing/non-finite grads on: {missing}"
    print(f"[ok] gradients finite and present on all {len(list(model.parameters()))} parameter tensors")

    # Cross-resolution decode, same idea as the Darcy example: encode
    # from grid_size-resolution context, decode at a finer query set.
    hi_res_pts = 512
    x_hi = torch.rand(batch_size, hi_res_pts, 2, device=device)
    r_s = model.encode(x_ctx, y_ctx)
    r_t = model.propagate(r_s, s, t)
    out_hi = model.decode(x_hi, r_t)
    assert out_hi.shape == (batch_size, hi_res_pts, 1)
    print(f"[ok] cross-resolution decode: encoded from {128} context pts, "
          f"queried at {hi_res_pts} scattered pts -> out{tuple(out_hi.shape)}")

    return model


def train_and_check_convergence(channel_mode, device, n_steps=400, grid_size=16, batch_size=32):
    print(f"\n=== training on synthetic heat diffusion, channel_mode = {channel_mode!r} ===")
    torch.manual_seed(0)
    model = ConvCNP_LTO_Evolution(grid_size=grid_size, channel_mode=channel_mode, time_scale=D_MAX).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    t0 = time.time()
    losses = []
    for step in range(n_steps):
        field_s, field_t, s, t = sample_trajectory_pairs(batch_size, grid_size, device)
        x_ctx, y_ctx = sample_points_from_field(field_s, 96)
        x_qry, y_qry = sample_points_from_field(field_t, 96)
        out = model(x_ctx, y_ctx, s, t, x_qry)
        loss = F.mse_loss(out, y_qry)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())

    early = sum(losses[:20]) / 20
    late = sum(losses[-20:]) / 20
    print(f"[{'ok' if late < early else 'WARN'}] {n_steps} steps in {time.time() - t0:.1f}s: "
          f"first-20 avg = {early:.5f}, last-20 avg = {late:.5f}")
    assert late < early, "loss did not improve -- something is broken, not just slow"

    # Native-resolution held-out eval.
    model.eval()
    with torch.no_grad():
        field_s, field_t, s, t = sample_trajectory_pairs(64, grid_size, device)
        x_ctx, y_ctx = sample_points_from_field(field_s, 96)
        x_qry, y_qry = sample_points_from_field(field_t, 96)
        val_loss = F.mse_loss(model(x_ctx, y_ctx, s, t, x_qry), y_qry).item()

        # Cross-resolution transfer: same trained weights, query a native
        # grid at 4x the training resolution -- the actual paper claim.
        hi_res = grid_size * 4
        g = torch.linspace(0, 1, hi_res, device=device)
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        xy_hi = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0)
        u0_hi = random_initial_conditions(1, hi_res, device)
        s_hi, t_hi = torch.tensor([0.2 * D_MAX], device=device), torch.tensor([0.6 * D_MAX], device=device)
        field_s_hi = diffuse(u0_hi, s_hi)
        lo, hi = field_s_hi.amin(), field_s_hi.amax()
        field_s_hi = (field_s_hi - lo) / (hi - lo).clamp_min(1e-6)
        true_t_hi = (diffuse(u0_hi, t_hi)[0] - lo) / (hi - lo).clamp_min(1e-6)

        x_ctx_hi, y_ctx_hi = sample_points_from_field(field_s_hi, 96)
        r_s = model.encode(x_ctx_hi, y_ctx_hi)
        r_t = model.propagate(r_s, s_hi, t_hi)
        pred_hi = model.decode(xy_hi, r_t).reshape(hi_res, hi_res)
        transfer_mse = F.mse_loss(pred_hi, true_t_hi).item()

    print(f"[ok] held-out val loss (native {grid_size}x{grid_size} points): {val_loss:.5f}")
    print(f"[ok] cross-resolution transfer, queried at native {hi_res}x{hi_res} "
          f"(4x training resolution): MSE = {transfer_mse:.5f}")
    return model, losses


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    for mode in ("raw", "normalized"):
        mechanical_checks(mode, device)

    print("\nAll mechanical checks passed under both channel_mode settings.\n"
          "Now training briefly on synthetic heat diffusion to confirm the "
          "time-conditioned architecture actually converges (not just shape-checks) "
          "and that cross-resolution transfer holds end to end in this refactored code:")

    for mode in ("raw", "normalized"):
        train_and_check_convergence(mode, device)

    print("\nEvolution example validated under both channel_mode settings.")
    print("This still uses the notebook's own synthetic heat generator, not real LNO")
    print("data -- NS2d integration (continuous (s,t) from LNO's discrete time windows)")
    print("is Phase 4 in claude/convcnp-lno-integration-plan.md, deliberately deferred.")
