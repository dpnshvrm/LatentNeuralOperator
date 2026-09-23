#!/bin/bash
#SBATCH --job-name eval-resolution-transfer-resaug-normalized-attn-aereconloss-extrap
#SBATCH --partition=nextlab200
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#
# Evaluates LTO_Darcy_resaug_normalized_attn_aereconloss_extrap (learned
# attention encoder + AE loss + context_frac_max=4.0) at both its latest
# (epoch 500) checkpoint and its tracked best.pt. Run this once
# lto_darcy_resaug_normalized_attn_aereconloss_extrap_job.sh finishes.
# Compare against LTO_Darcy_resaug_normalized_attn's own numbers
# (claude/convcnp-lno-integration-plan.md) at all 5 resolutions -- the
# AE loss is the genuinely new lever here (context_frac_max=4.0 isn't
# fixing a known break for this variant, see the training job's own
# comment).

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
# 2026-09-23: switched to /home/dverma/lno-conda -- the /scratch copy hit
# the same init_fs_encoding fatal error as the earlier lto-conda bug, so
# this env was rebuilt fresh under /home. See the training job's own
# comment for the full note.
source activate /home/dverma/lno-conda

python prepare_resolution_sweep.py

echo "=== LTO_Darcy_resaug_normalized_attn_aereconloss_extrap, latest (epoch 500) ==="
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_resaug_normalized_attn_aereconloss_extrap

echo ""
echo "=== LTO_Darcy_resaug_normalized_attn_aereconloss_extrap, best.pt ==="
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_resaug_normalized_attn_aereconloss_extrap --lto_epoch best
