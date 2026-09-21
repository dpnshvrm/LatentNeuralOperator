"""
find_lno_param_config.py

Grid search over LNO's capacity knobs (n_block, n_dim, n_head, n_mode) to
find a config landing close to a target parameter count -- written to
calibrate configs/LNO_time_NS2d_small.jsonc against LTO's NS2d
ae-reconloss checkpoint (1,126,211 params), per item 4 of the Aug/Sep 2026
Burgers/NS2d punch list (Dee asked for a param-matched LNO variant
alongside the natural-size Darcy-consistent one).

Instantiates the REAL module.model.LNO class for each candidate config and
counts parameters directly (sum(p.numel() for p in model.parameters() if
p.requires_grad)) rather than hand-deriving a parameter-count formula --
LNO's AttentionBlock/MLP shapes are simple but numerous enough that a
formula would be easy to get subtly wrong.

x_dim/y1_dim/y2_dim for NS2d are set from module/dataset.py's LNO_dataset
convention: x is the 2D grid coordinate (x_dim=2); NS2d is in the
"concat x onto y1" list, and the "_time" config's 10-frame sliding-window
history gives raw y1 width 10, so y1_dim = x_dim + 10 = 12;
model_attr["time"]=True forces the model's y2_dim to 1 (single-step-per-
autoregressive-step target), regardless of the dataset's own y2 last-dim
(10, the full unrolled horizon) -- see exp.py's train_time loop and
module/model.py's LNO.__init__ for where that forcing happens.

n_layer=2, attn="Attention_Vanilla", act="GELU" are held fixed, matching
every other LNO config in this project (LNO_Darcy.jsonc, LNO_time_NS2d.jsonc)
-- only the capacity knobs that actually move the needle on param count
(n_block, n_dim, n_head, n_mode) are swept.

Usage (run from LNO-PyTorch/ForwardProblem/, in the lno-conda env):
    python find_lno_param_config.py
    python find_lno_param_config.py --target 1126211 --top 15
"""

import argparse
from module.model import LNO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1_126_211,
                         help="Target param count. Default: LTO's NS2d ae-reconloss checkpoint (1,126,211).")
    parser.add_argument("--x_dim", type=int, default=2)
    parser.add_argument("--y1_dim", type=int, default=12)
    parser.add_argument("--y2_dim", type=int, default=1)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--attn", type=str, default="Attention_Vanilla")
    parser.add_argument("--act", type=str, default="GELU")
    parser.add_argument("--n_block_grid", type=str, default="2,3,4,6")
    parser.add_argument("--n_dim_grid", type=str, default="32,48,64,80,96,112,128,160")
    parser.add_argument("--n_head_grid", type=str, default="4,8")
    parser.add_argument("--n_mode_grid", type=str, default="64,128,256")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    model_attr = {"time": True}

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
