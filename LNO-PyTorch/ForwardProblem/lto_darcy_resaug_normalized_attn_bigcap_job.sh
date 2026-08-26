#!/bin/bash
#SBATCH --job-name lto-darcy-resaug-normalized-attn-bigcap
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
# Capacity scaling ON TOP of the attention encoder that already beat
# resaug_normalized by a wide margin (LTO_Darcy_resaug_normalized_attn:
# roughly halved in-range error, cut the 421 extrapolation error >5x, at
# matched parameter count -- see claude/convcnp-lno-integration-plan.md's
# 2026-08-26 result). This mirrors the exact playbook that worked on
# Advection: fix the mechanism first (there: bound the time gap via
# boundeddt; here: replace the fixed kernel with learned attention), THEN
# add capacity on top for a further multiplicative gain (there: combo,
# +~1.6x on top of boundeddt; here: this run, on top of the attention
# encoder).
#
# Critically different from the two PRIOR failed capacity attempts
# (bigcap_resaug, bigcap_resaug_normalized): those added capacity behind
# an encoder whose fixed-kernel locality bias was still the actual
# bottleneck, so more capacity just meant more overfitting risk on top of
# a broken mechanism. That bottleneck is gone now -- there's a real reason
# to expect capacity helps here, not just hope.
#
# configs/LTO_Darcy_resaug_normalized_attn_bigcap.jsonc: attn_dim=32,
# latent_channels=24, flow_hidden=96, decoder_hidden=192 (~240,594 params,
# close to bigcap_resaug_normalized's 255,603 -- same lr/weight_decay
# (3e-4 / 0.0005) that run established as right for this capacity scale).
#
# Own experiment/checkpoint dir, own master_port (12349) -- doesn't
# collide with any other Darcy job (LNO_Darcy 12341, LTO_Darcy 12342,
# LTO_Darcy_resaug 12343, LTO_Darcy_bigcap_resaug 12344,
# LTO_Darcy_resaug_normalized 12345, LTO_Darcy_resaug_normalized_dilated
# 12346, LTO_Darcy_bigcap_resaug_normalized 12347,
# LTO_Darcy_resaug_normalized_attn 12348).
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch lto_darcy_resaug_normalized_attn_bigcap_job.sh
# After it finishes:
#   sbatch eval_resolution_resaug_normalized_attn_bigcap_job.sh
#
# BEFORE SUBMITTING: verify use_attention_encoder/save_best are actually
# present in the Palmetto working copy (see the exp.py drift incident):
#   grep -n "save_best\|best_val_loss\|use_attention_encoder" exp.py module/utils.py module/convcnp_lto.py

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

bash scripts/LTO_Darcy_resaug_normalized_attn_bigcap.sh
