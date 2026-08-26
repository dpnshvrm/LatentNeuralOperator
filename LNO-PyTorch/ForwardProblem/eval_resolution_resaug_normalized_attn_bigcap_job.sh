#!/bin/bash
#SBATCH --job-name eval-resolution-transfer-resaug-normalized-attn-bigcap
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
# Evaluates LTO_Darcy_resaug_normalized_attn_bigcap at BOTH its latest
# (epoch 500) checkpoint and its tracked best.pt. Run this once
# lto_darcy_resaug_normalized_attn_bigcap_job.sh finishes. Compare against
# resaug_normalized_attn's own numbers (claude/convcnp-lno-integration-
# plan.md) at all 5 resolutions:
#   85: 0.04151, 106: 0.04140, 141: 0.04132, 211: 0.04125, 421: 0.04121

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

python prepare_resolution_sweep.py

echo "=== LTO_Darcy_resaug_normalized_attn_bigcap, latest (epoch 500) ==="
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_resaug_normalized_attn_bigcap

echo ""
echo "=== LTO_Darcy_resaug_normalized_attn_bigcap, best.pt ==="
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_resaug_normalized_attn_bigcap --lto_epoch best
