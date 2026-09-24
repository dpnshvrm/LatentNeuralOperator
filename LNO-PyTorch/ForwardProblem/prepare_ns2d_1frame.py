"""
prepare_ns2d_1frame.py

Re-slices the ALREADY-PREPARED NS2d_{train,val}.npy files (LNO's own
prepare.py output, {"x","y1","y2"} format -- x: (N,64,64,2) shared grid,
y1: (N,64,64,10) frames 0-9, y2: (N,64,64,10) frames 10-19, per
train_ns2d_fno_baseline.py's load_ns2d_split docstring) down to a
1-frame-context / 1-frame-target variant, for the 1-frame FNO/LNO ablation
requested 2026-09-24: does FNO/LNO's advantage over LTO shrink or vanish
once they're given the same single-state input LTO gets, instead of a
10-frame history window?

DOES NOT touch the raw .mat file or LNO's own prepare.py -- reuses the
SAME already-prepared trajectories, just re-cut. No new simulation data,
no re-running the original prepare.py step.

WHY FRAME 9 -> FRAME 10, NOT FRAME 0 -> FRAME 1: the existing "Direct"
single-step comparison (evaluate_ns2d_resolution_transfer_{fno,lno}.py,
and the FNO2d/LNO Direct numbers already in Table 2) is defined as
START_FRAME=9, target=frame 10 -- i.e. context ending at frame 9, single
un-chained prediction of frame 10. To isolate ONLY the context-window-
length variable (10 frames -> 1 frame) without also changing WHICH part
of the trajectory is being predicted, the 1-frame context must be frame 9
(the single frame immediately preceding the existing target), not frame 0
(the start of the trajectory) -- early- and late-trajectory dynamics are
not guaranteed to be equally hard, so anchoring at the wrong end would
silently confound window-length with trajectory-position.

WHY x IS BAKED INTO y1 HERE (NOT left for module/dataset.py to add):
module/dataset.py's LNO_dataset.__init__ only concatenates x onto y1 when
data_name is literally one of a hardcoded list
(["Darcy","Plasticity","Airfoil","Pipe","NS2d"]). Our new data_name
("NS2d_1frame") is deliberately NOT added to that list, to avoid editing
third-party LNO-PyTorch code at all -- instead this script does the same
concatenation itself, once, directly in the saved file, so LNO_dataset
loads y1 with x already included (final y1_dim = x_dim + 1 = 3, matching
find_lno_param_config.py's --y1_dim 3 grid search used to size
LNO_time_NS2d_1frame_small.jsonc). Concretely: x (N,64,64,2) concatenated
with the 1 context frame (N,64,64,1) on the last axis -> y1 (N,64,64,3).
y2 stays the single 1-frame target, untouched by this concatenation (only
y1 gets x appended, matching dataset.py's own behavior for "NS2d").

Verified against a synthetic-data smoke test before being run on real
data: reconstructing the full 20-frame trajectory as
np.concatenate([y1, y2], axis=-1) (the same convention
train_ns2d_fno_baseline.py's load_ns2d_split uses) and slicing
traj[..., 9:10] / traj[..., 10:11] exactly reproduces old y1's frame index
9 and old y2's frame index 0, and matches
evaluate_ns2d_resolution_transfer_fno.py's own
START_FRAME=9/N_TARGETS convention (first target = START_FRAME+1). Also
verified the x-concatenation step reproduces module/dataset.py's own
torch.cat((self.x, self.y1), dim=-1) exactly (same axis, same order:
coordinates first, then the frame).

OUTPUT: new files NS2d_1frame_{train,val}.npy, written alongside the
existing NS2d_{train,val}.npy -- does NOT overwrite them (every other
NS2d run in this project, including LTO's own training script and the
original NS2d LNO/FNO baselines, reads the original 10-frame files and
must keep working unchanged). Saved y1 here already includes x (shape
(N,64,64,3)); saved y2 is the single target frame (shape (N,64,64,1)).

Usage (run from LNO-PyTorch/ForwardProblem/, in the lno-conda env, after
the original NS2d_{train,val}.npy already exist):
    python prepare_ns2d_1frame.py --data_path <path to datas dir>
"""

import argparse
import os

import numpy as np


def resplice_split(data_path, split):
    in_path = os.path.join(data_path, f"NS2d_{split}.npy")
    d = np.load(in_path, allow_pickle=True).item()

    x = np.asarray(d["x"])    # (N, 64, 64, 2)
    y1 = np.asarray(d["y1"])  # (N, 64, 64, 10) -- frames 0-9, no coordinates yet
    y2 = np.asarray(d["y2"])  # (N, 64, 64, 10) -- frames 10-19, no coordinates

    assert x.ndim == 4 and x.shape[-1] == 2, f"unexpected x shape {x.shape}"
    assert y1.shape[-1] == 10, f"unexpected y1 shape {y1.shape} (expected 10 frames)"
    assert y2.shape[-1] == 10, f"unexpected y2 shape {y2.shape} (expected 10 frames)"
    assert x.shape[:3] == y1.shape[:3] == y2.shape[:3], (
        f"mismatched leading dims: x={x.shape}, y1={y1.shape}, y2={y2.shape}"
    )

    # Reconstruct the full 20-frame trajectory exactly as
    # train_ns2d_fno_baseline.py's load_ns2d_split does.
    traj = np.concatenate([y1, y2], axis=-1)  # (N, 64, 64, 20)
    assert traj.shape[-1] == 20

    START_FRAME = 9  # matches evaluate_ns2d_resolution_transfer_{fno,lno}.py
    ctx_frame = traj[..., START_FRAME:START_FRAME + 1]      # frame 9 only, (N,64,64,1)
    new_y2 = traj[..., START_FRAME + 1:START_FRAME + 2]     # frame 10 only, (N,64,64,1)

    # Sanity checks against the ORIGINAL arrays directly (not just the
    # reconstructed traj), so a bug in the concatenate step above can't
    # silently pass.
    assert np.array_equal(ctx_frame[..., 0], y1[..., 9]), "ctx_frame != original y1 frame index 9"
    assert np.array_equal(new_y2[..., 0], y2[..., 0]), "new_y2 != original y2 frame index 0"

    # Bake x onto y1 here (module/dataset.py won't do it for a data_name
    # it doesn't recognize) -- same axis/order as
    # LNO_dataset.__init__'s torch.cat((self.x, self.y1), dim=-1).
    new_y1 = np.concatenate([x, ctx_frame], axis=-1)  # (N, 64, 64, 3)

    assert new_y1.shape == (x.shape[0], 64, 64, 3)
    assert new_y2.shape == (x.shape[0], 64, 64, 1)
    # First 2 channels of new_y1 must be exactly x; last channel exactly the context frame.
    assert np.array_equal(new_y1[..., :2], x)
    assert np.array_equal(new_y1[..., 2:3], ctx_frame)

    out_path = os.path.join(data_path, f"NS2d_1frame_{split}.npy")
    np.save(out_path, {"x": x, "y1": new_y1, "y2": new_y2}, allow_pickle=True)
    print(f"{split}: wrote {out_path}  (x={x.shape}, y1={new_y1.shape} [x+frame9], "
          f"y2={new_y2.shape} [frame10], N={x.shape[0]} trajectories)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,
                         help="Directory containing NS2d_train.npy / NS2d_val.npy "
                              "(LNO's prepare.py output).")
    args = parser.parse_args()

    for split in ("train", "val"):
        in_path = os.path.join(args.data_path, f"NS2d_{split}.npy")
        if not os.path.exists(in_path):
            raise FileNotFoundError(
                f"{in_path} not found -- run `python prepare.py --data_name NS2d` first "
                f"(needs the raw NavierStokes_V1e-5_N1200_T20.mat file in --data_path)."
            )
        resplice_split(args.data_path, split)


if __name__ == "__main__":
    main()
