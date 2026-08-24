"""
Generate additional-resolution *evaluation-only* Darcy data for the
resolution-transfer sweep (train once at OBJ_RES=211 via prepare.py,
then evaluate the trained checkpoints at several OTHER resolutions here).

Only touches the val-source file (piececonst_r421_N1024_smooth2.mat) --
we're not retraining, just re-sampling the SAME underlying PDE solutions
at different resolutions to test whether accuracy holds up as resolution
changes. This is the core experiment behind the paper's discretization-
invariance claim (Theorem 2.3 / Corollary 2.4).

Run once, from the repo root, after the normal `prepare.py --data_name Darcy`:
    python prepare_resolution_sweep.py

Valid OBJ_RES values must divide 420 evenly (raw data is r421): 85, 106,
141, 211 (the training resolution -- already covered by Darcy_val.npy,
no need to regenerate), 421 (the raw native resolution, no downsampling).

Saves each as datas/darcy_sweep_r{R}.npy, a {"x", "y1", "y2"} dict --
deliberately NOT run through LNO_dataset's x-concat-onto-y1 /
normalization steps (those need to reuse the TRAIN set's normalizer, not
recompute fresh stats per resolution -- see evaluate_resolution_transfer.py,
which does that concatenation + normalization itself).

Does not import prepare.py (its argparse runs at module scope and would
fail without --data_name on the command line) -- load_Darcy is duplicated
here instead, kept identical to prepare.py's version.
"""

import os
import scipy.io as scio
import numpy as np
from module.setting import DATA_PATH

SRC_RES = 421
SWEEP_RESOLUTIONS = [85, 106, 141, 421]  # 211 already exists as Darcy_val.npy


def load_Darcy(path, src_res, obj_res):
    matdata = scio.loadmat(path)
    y1 = matdata['coeff']
    y1 = y1[:, ::(src_res-1)//(obj_res-1), ::(src_res-1)//(obj_res-1)][:, :obj_res, :obj_res]
    y2 = matdata['sol']
    y2 = y2[:, ::(src_res-1)//(obj_res-1), ::(src_res-1)//(obj_res-1)][:, :obj_res, :obj_res]
    x = np.reshape(np.dstack(np.meshgrid(np.linspace(0, 1, obj_res), np.linspace(0, 1, obj_res))), (obj_res, obj_res, 2))
    x = np.expand_dims(x, axis=0)
    x = np.repeat(x, y1.shape[0], axis=0)
    y1 = np.expand_dims(y1, axis=3)
    y2 = np.expand_dims(y2, axis=3)
    return x, y1, y2


if __name__ == "__main__":
    for obj_res in SWEEP_RESOLUTIONS:
        out_path = os.path.join(DATA_PATH, "darcy_sweep_r{}.npy".format(obj_res))
        if os.path.exists(out_path):
            print("Skipping r{} (already exists)".format(obj_res))
            continue
        x, y1, y2 = load_Darcy(
            os.path.join(DATA_PATH, "piececonst_r{}_N1024_smooth2.mat".format(SRC_RES)),
            SRC_RES, obj_res,
        )
        np.save(out_path, {"x": x, "y1": y1, "y2": y2})
        print("Saved r{}: x{} y1{} y2{}".format(obj_res, x.shape, y1.shape, y2.shape))
