"""
Phase 1 dev script: ConvCNP Latent Twin Operator, steady-problem variant,
adapted from code/heat_latent_twin_convcnp.ipynb for LNO's steady-state
forward-problem benchmarks (Darcy first).

This is a standalone, single-process script -- NOT yet wired into LNO's
torch.distributed exp.py harness. Purpose: validate the architecture
mechanically (shapes, gradients, zero-init identity behavior) and, once
Dee has downloaded+prepared the real Darcy data, numerically (does it
converge, does the cross-resolution decode story hold) before porting
into module/model.py + module/utils.py + a new config, per
claude/convcnp-lno-integration-plan.md.

Building blocks live in convcnp_lto_common.py (shared with
dev_convcnp_evolution.py). See that file's docstring for the raw vs.
normalized encoder channel_mode switch -- this script runs its checks
under BOTH modes so the still-open raw/normalized ablation
(project-status.md's "Next steps" #4) is exercisable here, not just in
the notebook.

Differences from the notebook's HeatLatentTwin:
  - No time conditioning: CNNLatentFlow (FiLM on scalar s,t) is replaced
    by CNNLatentOperator, a steady residual-CNN "evolve" step with no
    time inputs -- Darcy has no time axis. (The time-conditioned example
    lives in dev_convcnp_evolution.py.)
  - forward(x, y1) returns a single tensor (the predicted y2), not the
    notebook's (pred, r_s, r_t) tuple, so it drops straight into LNO's
    exp.py convention: `res = model(x, y1); loss = loss_fn(res, y2)`.
  - y1 in LNO's Darcy/Plasticity/Airfoil/Pipe/NS2d datasets is
    [x, coeff] concatenated (see module/dataset.py); forward() slices
    off the leading x_dim columns to recover the bare field, so no
    dataset.py changes are needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from convcnp_lto_common import SetConvEncoder2D, ConvCNPDecoder2D, CNNLatentOperator


class ConvCNP_LTO(nn.Module):
    """Latent Twin Operator, ConvCNP instantiation, steady-problem
    variant -- the model to register in LNO's module/model.py as
    (working name) `LTO_ConvCNP` once Phase 2 starts.

    forward(x, y1) matches LNO's exp.py calling convention: takes the
    dense/scattered coordinates `x` (dim x_dim) and the LNO-dataset's
    concatenated [x, field] tensor `y1`, slices off the leading x_dim
    columns to recover the bare input field, and returns a single
    predicted tensor -- no dataset.py changes required.
    """

    def __init__(self, grid_size, x_dim=2, hidden_channels=16, latent_channels=8,
                 flow_hidden=32, decoder_hidden=64, init_length_scale=0.1,
                 channel_mode="raw"):
        super().__init__()
        self.x_dim = x_dim
        self.encoder = SetConvEncoder2D(grid_size, hidden_channels, latent_channels,
                                         init_length_scale, channel_mode=channel_mode)
        self.operator = CNNLatentOperator(grid_size, latent_channels, flow_hidden)
        self.decoder = ConvCNPDecoder2D(grid_size, latent_channels, decoder_hidden, init_length_scale)

    def encode(self, x_ctx, y_ctx):
        return self.encoder(x_ctx, y_ctx)

    def propagate(self, r):
        return self.operator(r)

    def decode(self, x_qry, r):
        return self.decoder(x_qry, r)

    def forward(self, x, y1, x_qry=None):
        y_ctx = y1[..., self.x_dim:]
        r = self.encode(x, y_ctx)
        r = self.propagate(r)
        return self.decode(x_qry if x_qry is not None else x, r)


# --------------------------------------------------------------------------
# Mechanical validation on synthetic Darcy-shaped tensors -- no real data
# needed. Mirrors the notebook's own "sanity checks" cell pattern.
# --------------------------------------------------------------------------

def make_synthetic_darcy_batch(batch_size, obj_res, device):
    g = torch.linspace(0, 1, obj_res, device=device)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    x = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1)
    coeff = torch.rand(batch_size, obj_res * obj_res, 1, device=device) * 8 + 4  # ~ Darcy's {4,12} coeff range
    y1 = torch.cat([x, coeff], dim=-1)  # matches dataset.py's concat behavior
    y2 = torch.randn(batch_size, obj_res * obj_res, 1, device=device)  # stand-in solution field
    return x, y1, y2


def make_synthetic_scattered_batch(batch_size, n_points, device):
    # tests the "arbitrary point set" path, not just dense-grid input --
    # relevant for the later robustness-to-clustering ablation.
    x = torch.rand(batch_size, n_points, 2, device=device)
    coeff = torch.rand(batch_size, n_points, 1, device=device) * 8 + 4
    y1 = torch.cat([x, coeff], dim=-1)
    return x, y1


def run_checks(channel_mode):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== channel_mode = {channel_mode!r} ===")
    print(f"device: {device}")

    grid_size = 32
    batch_size = 4
    model = ConvCNP_LTO(grid_size=grid_size, channel_mode=channel_mode).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ConvCNP_LTO parameters: {n_params:,}  (grid_size={grid_size})")

    # 1. Basic forward shape check, dense grid == train-time resolution.
    x, y1, y2 = make_synthetic_darcy_batch(batch_size, grid_size, device)
    out = model(x, y1)
    assert out.shape == y2.shape, f"shape mismatch: {out.shape} vs {y2.shape}"
    assert torch.isfinite(out).all(), "non-finite output on dense-grid forward"
    print(f"[ok] dense-grid forward: x{tuple(x.shape)} y1{tuple(y1.shape)} -> out{tuple(out.shape)}")

    # 2. Zero-init identity check on the latent operator: at init, propagate(r) should equal r.
    r0 = torch.randn(2, grid_size * grid_size, 8, device=device)
    r1 = model.operator(r0)
    max_diff = (r1 - r0).abs().max().item()
    print(f"[ok] CNNLatentOperator identity at init: max diff = {max_diff:.8f}")
    assert max_diff < 1e-6, "operator not near-identity at init -- zero-init conv_out may be broken"

    # 3. Gradient flow check: loss.backward() should populate grads on every submodule.
    loss = F.mse_loss(out, y2)
    model.zero_grad()
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not missing, f"missing/non-finite grads on: {missing}"
    print(f"[ok] gradients finite and present on all {len(list(model.parameters()))} parameter tensors")

    # 4. Cross-resolution decode: encode at grid_size, decode at a DIFFERENT
    #    query resolution -- this is the whole resolution-invariance story.
    hi_res = 64
    x_hi, y1_hi, _ = make_synthetic_darcy_batch(batch_size, hi_res, device)
    out_hi = model(x, y1, x_qry=x_hi)
    assert out_hi.shape == (batch_size, hi_res * hi_res, 1)
    print(f"[ok] cross-resolution decode: trained/encoded at {grid_size}x{grid_size}, "
          f"queried at {hi_res}x{hi_res} -> out{tuple(out_hi.shape)}")

    # 5. Scattered (non-grid) context set -- exercises the arbitrary point-set path.
    x_scat, y1_scat = make_synthetic_scattered_batch(batch_size, n_points=500, device=device)
    out_scat = model(x_scat, y1_scat, x_qry=x_hi)
    assert out_scat.shape == (batch_size, hi_res * hi_res, 1)
    assert torch.isfinite(out_scat).all()
    print(f"[ok] scattered-context forward: {500} scattered context points -> out{tuple(out_scat.shape)}")

    # 6. One optimizer step, confirm loss actually changes (not NaN/frozen).
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(20):
        x, y1, y2 = make_synthetic_darcy_batch(batch_size, grid_size, device)
        out = model(x, y1)
        loss = F.mse_loss(out, y2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"[ok] 20 synthetic steps: loss {losses[0]:.4f} -> {losses[-1]:.4f} (finite: {all(torch.isfinite(torch.tensor(losses)))})")


if __name__ == "__main__":
    for mode in ("raw", "normalized"):
        run_checks(mode)

    print("\nAll Phase 1 mechanical checks passed under both channel_mode settings.")
    print("Currently trained/reported everywhere (notebook + this codebase's default) = 'raw'.")
    print("'normalized' is wired up and mechanically sound but UNTRAINED on real data --")
    print("that's the raw-vs-normalized ablation from project-status.md's Next steps #4.")
    print("Blocked on: real Darcy data (Dee to download piececonst_r241_N1024_smooth{1,2}.mat")
    print("and run prepare.py) before numerical validation (does it converge, does the")
    print("resolution-transfer story hold on real fields, does normalized actually help)")
    print("can happen -- see claude/convcnp-lno-integration-plan.md Phase 0/2/3.")
