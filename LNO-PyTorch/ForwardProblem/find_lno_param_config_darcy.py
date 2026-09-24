"""
find_lno_param_config_darcy.py

Grid search over LNO's capacity knobs (n_block, n_dim, n_head, n_mode) to
find a Darcy config landing close to a target parameter count -- the
Darcy counterpart to find_lno_param_config.py (which calibrated
configs/LNO_time_NS2d_small.jsonc against LTO's NS2d checkpoint).

Instantiates the REAL module.model.LNO class for each candidate config
and counts parameters directly (sum(p.numel() for p in
model.parameters() if p.requires_grad)) rather than hand-deriving a
parameter-count formula, for the same reason the NS2d script gives:
LNO's AttentionBlock/MLP shapes are simple but numerous enough that a
formula would be easy to get subtly wrong.

x_dim/y1_dim/y2_dim are set from module/dataset.py's LNO_dataset
convention for Darcy, verified against prepare.py's load_Darcy():
  - x is the 2D grid coordinate (x_dim=2).
  - Darcy is in the "concat x onto y1" list (dataset.py line ~86), and
    load_Darcy's raw y1 (the permeability field a(x)) is expanded to a
    single channel (np.expand_dims(y1, axis=3)), so y1_dim = x_dim + 1 = 3.
  - load_Darcy's raw y2 (the solution field u(x)) is likewise a single
    channel, so y2_dim = 1.
model_attr={"time": False}: Darcy is a static forward-map problem, not
NS2d's autoregressive time-stepping one, so the NS2d script's
model_attr={"time": True} does not apply here -- see exp.py's
get_model_data() call, which only forces y2_dim=1 for the time=True
case (already true here anyway since Darcy's own y2_dim is 1).

n_layer=2, attn="Attention_Vanilla", act="GELU" are held fixed, matching
LNO_Darcy.jsonc (the project's only other Darcy LNO config) -- only the
capacity knobs that actually move the needle on param count (n_block,
n_dim, n_head, n_mode) are swept.

Target: our published LTO Darcy configs are far smaller than NS2d's
(29,139 params for the plain-kernel encoder, 240,594 for the
attn+bigcap "ours" headline variant) and far smaller than LNO_Darcy's
own native 762,113 -- so unlike the NS2d search, the grids below are
widened toward the low end (n_dim down to 16, n_block down to 1, n_mode
down to 32) to give the search room to reach these targets. If the
closest candidate for your chosen target is still not close enough,
widen --n_dim_grid/--n_block_grid/--n_mode_grid further on the command
line rather than editing this file.

Usage (run from LNO-PyTorch/ForwardProblem/, in the lno-conda env):
    # Match LTO's plain-kernel / attn-only Darcy configs (default):
    python find_lno_param_config_darcy.py
    # Match LTO's headline attn+bigcap "ours" config instead:
    python find_lno_param_config_darcy.py --target 240594

RESULT (2026-09-24, verified against the real module.model.LNO class):
n_block=3, n_dim=26, n_head=2, n_mode=128 -> 29,171 params (+0.11% over
the plain-kernel target 29,139, +0.96% over the attn-only target
28,894 -- one config serves both). Already written into
configs/LNO_Darcy_matched.jsonc; rerun this script only if you want a
different target or a finer search.
"""

import argparse
from module.model import LNO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=29_139,
                         help="Target param count. Default: LTO's Darcy "
                              "plain-kernel baseline (29,139; the attn-only "
                              "baseline at 28,894 is close enough that the "
                              "same match serves both). Pass --target 240594 "
                              "to match LTO's attn+bigcap 'ours' config "
                              "instead.")
    parser.add_argument("--x_dim", type=int, default=2)
    parser.add_argument("--y1_dim", type=int, default=3)
    parser.add_argument("--y2_dim", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--attn", type=str, default="Attention_Vanilla")
    parser.add_argument("--act", type=str, default="GELU")
    parser.add_argument("--n_block_grid", type=str, default="1,2,3,4")
    parser.add_argument("--n_dim_grid", type=str, default="16,24,32,48,64,80,96,112,128")
    parser.add_argument("--n_head_grid", type=str, default="2,4,8")
    parser.add_argument("--n_mode_grid", type=str, default="32,64,128,256")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    model_attr = {"time": False}

    n_block_grid = [int(v) for v in args.n_block_grid.split(",")]
    n_dim_grid = [int(v) for v in args.n_dim_grid.split(",")]
    n_head_grid = [int(v) for v in args.n_head_grid.split(",")]
    n_mode_grid = [int(v) for v in args.n_mode_grid.split(",")]

    candidates = []
    for n_block in n_block_grid:
        for n_dim in n_dim_grid:
            for n_head in n_head_grid:
                if n_dim % n_head != 0:
                    continue
                for n_mode in n_mode_grid:
                    model = LNO(n_block=n_block, n_mode=n_mode, n_dim=n_dim, n_head=n_head,
                                n_layer=args.n_layer, x_dim=args.x_dim, y1_dim=args.y1_dim,
                                y2_dim=args.y2_dim, attn=args.attn, act=args.act, model_attr=model_attr)
                    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    candidates.append((n_params, n_block, n_dim, n_head, n_mode))

    candidates.sort(key=lambda c: abs(c[0] - args.target))
    print(f"Target: {args.target:,} params\n")
    print(f"{'params':>12} | {'n_block':>7} | {'n_dim':>5} | {'n_head':>6} | {'n_mode':>6} | diff")
    print("-" * 60)
    for n_params, n_block, n_dim, n_head, n_mode in candidates[:args.top]:
        print(f"{n_params:>12,} | {n_block:>7} | {n_dim:>5} | {n_head:>6} | {n_mode:>6} | {n_params - args.target:+,}")


if __name__ == "__main__":
    main()
