#!/bin/bash
#SBATCH --job-name lto-darcy-resaug-normalized-aereconloss-extrap
#SBATCH --partition=nextlab200
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#
# Two changes on top of our locked small-cap, plain-Gaussian-kernel Darcy
# baseline (LTO_Darcy_resaug_normalized, 29,139 params), both new
# 2026-09-23, neither previously tried on Darcy:
#
# 1. AE reconstruction loss (config: loss.name="rL2_ae"). Ports the
#    train_burgers_pdebench.py / train_ns2d_evolution.py
#    --recon_loss_weight mechanism (encode+decode round-trip, no
#    propagate() call, auxiliary loss term) into this LNO/exp.py harness
#    -- see module/loss.py's RelLpLossWithRecon and
#    module/convcnp_lto.py's forward() docstrings for the full mechanism
#    and the design note on why this needed a small (additive,
#    backward-compatible) exp.py change to pass y2 into forward().
#
# 2. context_frac_max extended 1.0 -> 4.0 (config field). THIS is the
#    checkpoint that actually needs it: resaug_normalized is still
#    genuinely broken at the super-native eval point (resolution 421:
#    0.218 rL2, 25.6x gap to LNO, its worst point on the curve -- see
#    claude/convcnp-lno-integration-plan.md's 2026-08-23 result). 4.0,
#    not 2.0 -- context_frac is a POINT-COUNT fraction of the native
#    grid, and resolution 421's point count ratio is (421*421)/(211*211)
#    = 3.98, not the resolution ratio 421/211 = 2.0 (see the config's
#    own comment and train_ns2d_evolution.py's make_context_query()
#    docstring for the same distinction, caught there first). Uses the
#    exact same bilinear-interpolation densification method as that
#    NS2d fix, adapted to Darcy's batched coordinate convention -- see
#    module/convcnp_lto.py's _densify_context()/_augment_context().
#
# Smoke-tested locally before this run (synthetic square-grid data,
# small grid_size, both encoder variants, with and without y2, forced
# super-native context draws, backward() through the combined loss) --
# see claude/convcnp-lno-integration-plan-part4.md's 2026-09-23 section.
#
# Own experiment/checkpoint dir (LTO_Darcy_resaug_normalized_aereconloss_extrap),
# own master_port (12350) -- doesn't collide with any other Darcy job
# (LNO_Darcy 12341 ... LTO_Darcy_resaug_normalized_attn 12348,
# LTO_Darcy_resaug_normalized_attn_bigcap 12349).
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch lto_darcy_resaug_normalized_aereconloss_extrap_job.sh
# After it finishes, evaluate resolution transfer against it:
#   sbatch eval_resolution_resaug_normalized_aereconloss_extrap_job.sh
#
# BEFORE SUBMITTING: verify this exp.py / module/utils.py / module/loss.py /
# module/convcnp_lto.py push actually landed on Palmetto (see the earlier
# "exp.py drift incident" note in claude/convcnp-lno-integration-plan.md):
#   grep -n "needs_y2\|rL2_ae\|RelLpLossWithRecon\|compute_recon_loss" exp.py module/utils.py module/loss.py module/convcnp_lto.py
#
# CONDA_ENV_PATH note (UPDATED 2026-09-23): this job actually hit the
# same init_fs_encoding/codec fatal-error as the earlier /scratch/dverma/
# lto-conda bug -- this time on /scratch/dverma/lno-conda, which every
# prior Darcy job had used successfully. Same root cause pattern (env
# under /scratch, not /home). Fixed the same way: env rebuilt fresh at
# /home/dverma/lno-conda (Python 3.9, same package list as the original --
# pip install torch numpy scipy einops jsmin matplotlib tqdm tensorboard).
# Every OTHER already-run Darcy job script still points at the old
# /scratch/dverma/lno-conda path and was deliberately left untouched
# (doesn't need to rerun) -- only this job and its sibling
# (lto_darcy_resaug_normalized_attn_aereconloss_extrap_job.sh) plus their
# two eval job scripts were updated, since none of the four had
# successfully run yet.

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /home/dverma/lno-conda

bash scripts/LTO_Darcy_resaug_normalized_aereconloss_extrap.sh
