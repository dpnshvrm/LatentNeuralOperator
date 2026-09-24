#!/bin/bash
#SBATCH --job-name eval-resolution-transfer-resaug-normalized-aereconloss-extrap-fullnorm
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
# Identical to eval_resolution_resaug_normalized_aereconloss_extrap_job.sh,
# except --lto_exp points at LTO_Darcy_resaug_normalized_aereconloss_extrap_
# fullnorm (channel_mode fully_normalized instead of normalized -- see
# lto_darcy_resaug_normalized_aereconloss_extrap_fullnorm_job.sh). No
# script changes needed -- evaluate_resolution_transfer.py already takes
# an arbitrary --lto_exp. Run this once
# lto_darcy_resaug_normalized_aereconloss_extrap_fullnorm_job.sh finishes.
# Compare against BOTH LTO_Darcy_resaug_normalized_aereconloss_extrap's own
# numbers (the "normalized"-only sibling, running in parallel) and
# LTO_Darcy_resaug_normalized's numbers (claude/convcnp-lno-integration-
# plan.md) at all 5 resolutions, especially resolution 421 (0.218 rL2,
# 25.6x gap to LNO on the plain resaug_normalized checkpoint -- the point
# this whole fullnorm track is meant to fix).

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /home/dverma/lno-conda

python prepare_resolution_sweep.py

echo "=== LTO_Darcy_resaug_normalized_aereconloss_extrap_fullnorm, latest (epoch 500) ==="
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_resaug_normalized_aereconloss_extrap_fullnorm

echo ""
echo "=== LTO_Darcy_resaug_normalized_aereconloss_extrap_fullnorm, best.pt ==="
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_resaug_normalized_aereconloss_extrap_fullnorm --lto_epoch best
