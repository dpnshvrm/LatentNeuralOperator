"""
evaluate_ns2d.py

Direct-literature-comparison eval for ConvCNP_LTO's NS2d time-evolution
result (see the "NS2d" section of claude/convcnp-lno-integration-plan.md).

Unlike the Darcy/LNO comparison, this does NOT retrain or evaluate an LNO
checkpoint -- LNO's own paper already reports a number for this exact
benchmark (Table 1, assets/Forward-1.png: 8.45e-2 relative L2), so that
published number is cited directly below.

This script reproduces LNO's own task convention exactly: module/dataset.py's
NS2d loader (unmodified), and the same T_in=10 input frames -> T=10-step
autoregressive rollout used by exp.py's val_time (crucially: the rollout
here feeds the model's OWN prediction back into the sliding window at each
step, not ground truth -- val_time does this, train_time does NOT, see
exp.py). Same rL2 loss definition (module/loss.py's RelLpLoss) as the
config's "loss": {"name": "rL2"}. This makes the number this script prints
apples-to-apples with LNO's published one, not a different task dressed up
to look comparable.

Per this project's standing discipline (see claude/convcnp-lno-integration-plan.md's
training-log-vs-eval-script caution -- the LNO harness's own printed "Val
Loss Full" has previously NOT matched what an independent eval script
computes for at least one other checkpoint): don't just read the training
log's last "Best Val Loss Full" number and report it as the final result --
run this script against the saved checkpoint and use ITS number.

Run once LTO_time_NS2d has trained (or has at least a best.pt / numbered
checkpoint):
    python evaluate_ns2d.py --lto_exp LTO_time_NS2d --lto_epoch best
"""

import argparse
import glob
import os

import torch

from module.dataset import LNO_dataset
from module.loss import RelLpLoss
from module.model import ConvCNP_LTO
from module.setting import EXP_PATH
from module.utils import Configuration, get_num_params

# LNO paper Table 1 (assets/Forward-1.png), reported directly -- NOT
# retrained or reproduced here. See the module docstring above.
LNO_PUBLISHED_NS2D_RL2 = 0.0845

_parser = argparse.ArgumentParser()
_parser.add_argument("--lto_exp", default="LTO_time_NS2d",
                      help="LTO experiment/checkpoint dir name under EXP_PATH.")
_parser.add_argument("--lto_config", default=None,
                      help="LTO config name (without .jsonc), defaults to --lto_exp.")
_parser.add_argument("--lto_epoch", type=str, default=None,
                      help="Integer epoch number, or the literal 'best' to load best.pt "
                           "(the lowest-val_loss_full checkpoint tracked by exp.py's "
                           "train_time -- see module/utils.py's Checkpoint.save_best()). "
                           "Defaults to the latest numbered epoch.")
_parser.add_argument("--val_batch_size", type=int, default=4)
_args = _parser.parse_args()
LTO_EXP_NAME = _args.lto_exp
LTO_CONFIG_NAME = _args.lto_config or _args.lto_exp
LTO_EPOCH = _args.lto_epoch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                    "No best.pt in {} -- use an explicit --lto_epoch <N> instead.".format(ckpt_dir))
            return wanted, "best"
        wanted = os.path.join(ckpt_dir, "{}.pt".format(epoch))
        if wanted not in ckpts:
            available = sorted(int(os.path.splitext(os.path.basename(p))[0])
                                for p in ckpts if os.path.basename(p) != "best.pt")
            raise FileNotFoundError(
                "No checkpoint for epoch {} in {} -- available epochs: {}".format(epoch, ckpt_dir, available))
        return wanted, epoch
    numbered = [p for p in ckpts if os.path.basename(p) != "best.pt"]
    if not numbered:
        raise FileNotFoundError("Only best.pt found in {} -- pass --lto_epoch best explicitly.".format(ckpt_dir))
    latest = max(numbered, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    return latest, int(os.path.splitext(os.path.basename(latest))[0])


def strip_module_prefix(state_dict):
    # Checkpoints were saved from a DistributedDataParallel-wrapped model
    # ("module." key prefix) -- this script has no distributed process
    # group, so strip it before loading into a plain model.
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def build_lto(config):
    return ConvCNP_LTO(config.model.grid_size, 2, config.model.hidden_channels, config.model.latent_channels,
                        config.model.flow_hidden, config.model.decoder_hidden, config.model.init_length_scale,
                        config.model.channel_mode,
                        use_dilated=config.model.get("use_dilated", False),
                        use_attention_encoder=config.model.get("use_attention_encoder", False),
                        attn_dim=config.model.get("attn_dim", 32),
                        y_dim=config.model.get("y_dim", 1)).to(DEVICE)


@torch.no_grad()
def evaluate_rollout(model, val_dataloader):
    model.eval()
    loss_fn = RelLpLoss(p=2)
    step_losses, full_losses = [], []

    for x, y1, y2 in val_dataloader:
        x = x.to(DEVICE)
        x = torch.reshape(x, (x.shape[0], -1, x.shape[-1]))
        y1 = y1.to(DEVICE)
        y1 = torch.reshape(y1, (y1.shape[0], -1, y1.shape[-1]))
        y2 = y2.to(DEVICE)
        y2 = torch.reshape(y2, (y2.shape[0], -1, y2.shape[-1]))

        T, step = 10, 1
        loss = 0.0
        pred_full = None
        for t in range(0, T, step):
            gt = y2[..., t:t + step]
            pred_step = model(x, y1)
            loss = loss + loss_fn(pred_step, gt)
            pred_full = pred_step if t == 0 else torch.cat((pred_full, pred_step), -1)
            # Autoregressive: feed the model's OWN prediction back into the
            # sliding window (not ground truth) -- matches exp.py's
            # val_time exactly. (train_time uses teacher forcing with
            # `gt` instead -- that's the training-time-only shortcut, not
            # what's used to report an accuracy number.)
            y1 = torch.cat((y1[..., :2], y1[..., 2 + step:], pred_step), dim=-1)

        step_losses.append(loss.item())
        full_losses.append(loss_fn(pred_full, y2).item())

    return sum(step_losses) / len(step_losses), sum(full_losses) / len(full_losses)


if __name__ == "__main__":
    lto_config = Configuration(os.path.join("configs", "{}.jsonc".format(LTO_CONFIG_NAME)))
    lto_ckpt, lto_epoch = latest_checkpoint(LTO_EXP_NAME, epoch=LTO_EPOCH)
    print("LTO checkpoint: epoch {} ({})".format(lto_epoch, lto_ckpt))

    lto_model = build_lto(lto_config)
    lto_model.load_state_dict(strip_module_prefix(torch.load(lto_ckpt, map_location=DEVICE)))
    print("Parameter count: LTO = {:,}".format(get_num_params(lto_model)))

    val_dataset = LNO_dataset("NS2d", "val")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=_args.val_batch_size, shuffle=False, drop_last=False)

    step_rl2, full_rl2 = evaluate_rollout(lto_model, val_dataloader)
    print("\nFull 10-step autoregressive rollout, native 64x64, val split (200 held-out trajectories):")
    print("  Val rL2 (step-summed over 10 steps, matches training's 'Val Loss Step'): {:.5f}".format(step_rl2))
    print("  Val rL2 (full rollout vs. full 10-frame target, matches training's 'Val Loss Full'"
          " -- THIS is the number to compare): {:.5f}".format(full_rl2))
    print("\nLNO published NS2d rL2 (paper Table 1, assets/Forward-1.png -- NOT retrained here): {:.4f}".format(
        LNO_PUBLISHED_NS2D_RL2))
    print("Ratio (LTO / LNO published): {:.2f}x".format(full_rl2 / LNO_PUBLISHED_NS2D_RL2))
