"""
Resolution-transfer evaluation: train once at OBJ_RES=211 (both LNO_Darcy
and LTO_Darcy), then evaluate the TRAINED, FROZEN checkpoints at several
resolutions never seen in training. This is the actual paper experiment
behind Theorem 2.3 / Corollary 2.4 (discretization invariance) -- not
just a training-loss sanity check.

Run once both training jobs have finished (or at least saved a
checkpoint) and prepare_resolution_sweep.py has been run:
    python evaluate_resolution_transfer.py

Prints a table of rL2 error vs. resolution for both models, plus their
parameter counts (the efficiency side of the comparison -- see
claude/convcnp-lno-integration-plan.md, LTO has ~25x fewer params).

Note on LNO: model.py's plain `LNO` class uses the SAME projected `x` for
both the encode and decode side internally (no separate query
positions), so what's tested here for LNO is "does it generalize to an
entirely new resolution presented on both the input and output side" --
not the fully decoupled encode-at-A/query-at-B transfer that
ConvCNP_LTO supports natively via a separate x_qry. See the integration
plan for why: `LNO_triple` (decoupled positions) is a different
architecture with different weights, not evaluable from this checkpoint.
"""

import argparse
import glob
import os

import numpy as np
import torch

from module.dataset import LNO_dataset
from module.loss import RelLpLoss
from module.model import LNO, ConvCNP_LTO
from module.setting import DATA_PATH, EXP_PATH
from module.utils import Configuration, get_num_params

RESOLUTIONS = [85, 106, 141, 211, 421]

# --lto_exp / --lto_config let this same script evaluate a different LTO
# experiment/config -- e.g. LTO_Darcy_resaug (the context-size-augmentation
# fix, see claude/convcnp-lno-integration-plan.md) -- without duplicating
# the whole script. Defaults reproduce the original LTO_Darcy vs LNO_Darcy
# comparison exactly.
_parser = argparse.ArgumentParser()
_parser.add_argument("--lto_exp", default="LTO_Darcy",
                      help="LTO experiment/checkpoint dir name under EXP_PATH, e.g. LTO_Darcy_resaug.")
_parser.add_argument("--lto_config", default=None,
                      help="LTO config name (without .jsonc), defaults to --lto_exp if not given -- "
                           "matches this repo's convention of naming the config after the experiment.")
_parser.add_argument("--lto_epoch", type=str, default=None,
                      help="Evaluate a SPECIFIC saved checkpoint instead of the latest epoch -- either "
                           "an integer epoch number (must match a checkpoint actually saved, every "
                           "model_save_interval_epoch) or the literal 'best' to load best.pt, the "
                           "lowest-val-loss checkpoint tracked every epoch by exp.py's train()/"
                           "train_time() (see module/utils.py's Checkpoint.save_best(), added "
                           "2026-08-24 -- only present for experiments trained after that date). "
                           "Defaults to None = latest numbered epoch, the original behavior.")
_args = _parser.parse_args()
LTO_EXP_NAME = _args.lto_exp
LTO_CONFIG_NAME = _args.lto_config or _args.lto_exp
LTO_EPOCH = _args.lto_epoch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_BATCH_SIZE = 2  # kept small deliberately -- LTO's SetConv is O(grid_size^2 * n_points);
                     # at res=421 (177k points) even batch_size=2 is a multi-GB pairwise tensor.


def latest_checkpoint(exp_name, epoch=None):
    ckpt_dir = os.path.join(EXP_PATH, exp_name, "checkpoint")
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found in {}".format(ckpt_dir))
    if epoch is not None:
        if str(epoch) == "best":
            wanted = os.path.join(ckpt_dir, "best.pt")
            if wanted not in ckpts:
                raise FileNotFoundError(
                    "No best.pt in {} -- this experiment predates the best-checkpoint tracking added "
                    "2026-08-24 (module/utils.py's Checkpoint.save_best()), or hasn't run since. Use "
                    "an explicit --lto_epoch <N> instead, or retrain.".format(ckpt_dir))
            return wanted, "best"
        wanted = os.path.join(ckpt_dir, "{}.pt".format(epoch))
        if wanted not in ckpts:
            # best.pt (if present) isn't a numbered epoch -- exclude it here
            # so this doesn't crash trying int("best").
            available = sorted(int(os.path.splitext(os.path.basename(p))[0])
                                for p in ckpts if os.path.basename(p) != "best.pt")
            raise FileNotFoundError(
                "No checkpoint for epoch {} in {} -- available epochs: {}".format(epoch, ckpt_dir, available))
        return wanted, epoch
    # "latest" means latest NUMBERED epoch -- best.pt is never auto-picked
    # here even if present, since it isn't ordered by epoch number.
    numbered = [p for p in ckpts if os.path.basename(p) != "best.pt"]
    if not numbered:
        raise FileNotFoundError(
            "Only best.pt found in {} -- pass --lto_epoch best explicitly.".format(ckpt_dir))
    latest = max(numbered, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    epoch = int(os.path.splitext(os.path.basename(latest))[0])
    return latest, epoch


def strip_module_prefix(state_dict):
    # Checkpoints were saved from a DistributedDataParallel-wrapped model,
    # whose state_dict keys are prefixed "module." -- strip it to load
    # into a plain (non-DDP) model here, since this script has no
    # distributed process group.
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def load_resolution_data(res):
    if res == 211:
        # the resolution actually trained on -- reuse the existing val split
        dataset = np.load(os.path.join(DATA_PATH, "Darcy_val.npy"), allow_pickle=True).tolist()
    else:
        dataset = np.load(os.path.join(DATA_PATH, "darcy_sweep_r{}.npy".format(res)), allow_pickle=True).tolist()
    x = torch.tensor(dataset["x"]).float()
    y1 = torch.tensor(dataset["y1"]).float()
    y1 = torch.cat((x, y1), dim=-1)  # matches LNO_dataset's Darcy convention
    y2 = torch.tensor(dataset["y2"]).float()
    return x, y1, y2


def train_normalizer():
    # Reconstruct the SAME normalizer stats used at training time, from
    # the train split -- NOT recomputed per-resolution, or this wouldn't
    # be testing the trained model's actual generalization.
    return LNO_dataset("Darcy", "train").get_normalizer()


@torch.no_grad()
def evaluate(model, normalizer, x, y1, y2, batch_size=EVAL_BATCH_SIZE):
    model.eval()
    loss_fn = RelLpLoss(p=2)
    x = normalizer.apply_x(x.to(DEVICE))
    y1 = normalizer.apply_y1(y1.to(DEVICE))
    y2 = y2.to(DEVICE)  # keep raw -- un-normalize predictions instead, matching module/utils.py's val()

    losses = []
    for i in range(0, x.shape[0], batch_size):
        xb = torch.reshape(x[i:i+batch_size], (x[i:i+batch_size].shape[0], -1, x.shape[-1]))
        y1b = torch.reshape(y1[i:i+batch_size], (y1[i:i+batch_size].shape[0], -1, y1.shape[-1]))
        y2b = torch.reshape(y2[i:i+batch_size], (y2[i:i+batch_size].shape[0], -1, y2.shape[-1]))
        res = model(xb, y1b)
        res = normalizer.apply_y2(res, inverse=True)
        losses.append(loss_fn(res, y2b).item())
    return sum(losses) / len(losses)


def build_lno(config):
    return LNO(config.model.n_block, config.model.n_mode, config.model.n_dim, config.model.n_head,
               config.model.n_layer, 2, 3, 1, config.model.attn, config.model.act, {"time": False}).to(DEVICE)


def build_lto(config):
    # use_dilated must be read from the config and passed through, or a
    # dilated checkpoint's state_dict (operator.dilated_convs.*) won't
    # match a model built with the default plain conv_mid -- exactly the
    # "Missing/Unexpected key(s)" error this fixes. .get() defaults to
    # False so every non-dilated config (the vast majority) is unaffected.
    return ConvCNP_LTO(config.model.grid_size, 2, config.model.hidden_channels, config.model.latent_channels,
                        config.model.flow_hidden, config.model.decoder_hidden, config.model.init_length_scale,
                        config.model.channel_mode,
                        use_dilated=config.model.get("use_dilated", False)).to(DEVICE)


if __name__ == "__main__":
    normalizer = train_normalizer()

    lno_config = Configuration(os.path.join("configs", "LNO_Darcy.jsonc"))
    lto_config = Configuration(os.path.join("configs", "{}.jsonc".format(LTO_CONFIG_NAME)))

    lno_ckpt, lno_epoch = latest_checkpoint("LNO_Darcy")
    lto_ckpt, lto_epoch = latest_checkpoint(LTO_EXP_NAME, epoch=LTO_EPOCH)
    print("LNO checkpoint: epoch {} ({})".format(lno_epoch, lno_ckpt))
    print("LTO checkpoint: epoch {} ({})".format(lto_epoch, lto_ckpt))

    lno_model = build_lno(lno_config)
    lno_model.load_state_dict(strip_module_prefix(torch.load(lno_ckpt, map_location=DEVICE)))

    lto_model = build_lto(lto_config)
    lto_model.load_state_dict(strip_module_prefix(torch.load(lto_ckpt, map_location=DEVICE)))

    print("\nParameter counts: LNO = {:,}   LTO = {:,}\n".format(
        get_num_params(lno_model), get_num_params(lto_model)))

    print("{:>10}  {:>12}  {:>12}".format("Res", "LNO rL2", "LTO rL2"))
    for res in RESOLUTIONS:
        x, y1, y2 = load_resolution_data(res)
        lno_err = evaluate(lno_model, normalizer, x, y1, y2)
        lto_err = evaluate(lto_model, normalizer, x, y1, y2)
        print("{:>10}  {:>12.5f}  {:>12.5f}".format(res, lno_err, lto_err))
