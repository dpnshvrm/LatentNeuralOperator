"""
Diagnostic for the Darcy resolution-transfer regression: LTO's rL2 at
resolutions away from the native training resolution (211) got MUCH worse
between epoch 50 (0.213-0.929 range) and epoch 500 (0.108-18.95 range),
even though the epoch-500 checkpoint is far better AT the native
resolution (0.213 -> 0.108). That specific pattern -- native-resolution
error drops while error at other resolutions explodes, worst at the
SPARSEST resolutions (85: 18.95) and much milder at the densest (421:
0.78) -- is the signature of the SetConv encoder/decoder's LEARNED
length-scale (`log_length_scale`) shrinking over training to fit the
211-resolution point spacing tightly. A too-small length scale means at
sparser resolutions (85, 106, 141 -- fewer points, larger gaps between
them) most latent grid cells see near-zero density (few/no context
points fall within a few length-scales), so encoder output collapses
toward near-zero/degenerate features the decoder was never trained to
handle -- consistent with the catastrophic (>1, i.e. worse than
predicting zero) errors at low resolution. At 421 (denser than training,
smaller gaps) there's no such starvation, hence the much smaller (though
still 421 > 211 native by ~7x) error there.

This script does NOT need a GPU, the dataset, or LNO's harness code --
it just loads each saved LTO_Darcy checkpoint's state_dict directly and
reads the two scalar length-scale parameters (encoder's and decoder's),
converting log_length_scale -> length_scale via exp(). Run it from the
LatentNeuralOperator/ForwardProblem directory (wherever the checkpoints
actually live):

    python inspect_lto_length_scales.py

For reference, coordinate spacing at each resolution tested (domain is
[0,1]^2, so spacing = 1/(res-1)):
    res=85:  spacing ~0.0119   res=106: spacing ~0.00952
    res=141: spacing ~0.00714  res=211: spacing ~0.00476 (native/training)
    res=421: spacing ~0.00238
And the LATENT grid (grid_size=64) spacing: 1/63 ~ 0.01587.

If the encoder's length scale has shrunk to something close to or below
211's spacing (~0.00476), that's the smoking gun: it means the model
learned a kernel width barely wide enough to cover the training
resolution's own point density, with no margin for sparser test
resolutions (whose spacing is 1.5x-2.5x larger) to still fall within
reach of the kernel.
"""

import glob
import os
import math

import torch

CHECKPOINT_DIR = os.path.join("experiments", "LTO_Darcy", "checkpoint")


def strip_module_prefix(state_dict):
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def main():
    ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
    ckpts = sorted(ckpts, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    print(f"{'Epoch':>8} | {'Encoder ell':>12} | {'Decoder ell':>12}")
    print("-" * 40)
    for ckpt_path in ckpts:
        epoch = int(os.path.splitext(os.path.basename(ckpt_path))[0])
        sd = strip_module_prefix(torch.load(ckpt_path, map_location="cpu"))
        enc_ell = math.exp(sd["encoder.log_length_scale"].item())
        dec_ell = math.exp(sd["decoder.log_length_scale"].item())
        print(f"{epoch:>8} | {enc_ell:>12.6f} | {dec_ell:>12.6f}")

    print()
    print("Reference coordinate spacing (domain [0,1]^2, spacing = 1/(res-1)):")
    for res in [85, 106, 141, 211, 421]:
        tag = "  <- native/training resolution" if res == 211 else ""
        print(f"  res={res:>4}: spacing ~{1/(res-1):.5f}{tag}")
    print(f"  latent grid (grid_size=64): spacing ~{1/63:.5f}")
    print()
    print("If the final epoch's encoder/decoder length scale is close to or below")
    print("the res=211 spacing (~0.00476), that supports the hypothesis: the kernel")
    print("shrank to fit the training resolution tightly, leaving no margin for the")
    print("sparser test resolutions (85/106/141), which is what the resolution-")
    print("transfer eval's catastrophic low-resolution errors would be explained by.")


if __name__ == "__main__":
    main()
