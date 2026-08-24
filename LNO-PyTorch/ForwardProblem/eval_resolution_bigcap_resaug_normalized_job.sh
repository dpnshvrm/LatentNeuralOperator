#!/bin/bash
#SBATCH --job-name eval-resolution-transfer-bigcap-resaug-normalized
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
# Evaluates LTO_Darcy_bigcap_resaug_normalized at BOTH its latest (epoch
# 500) checkpoint and its tracked best.pt (lowest val loss seen during
# training) -- run this once lto_darcy_bigcap_resaug_normalized_job.sh
# finishes. Comparing the two tells us whether this run also overfits
# past an earlier optimum (like the dilated variant did) or whether the
# lower LR/higher weight decay kept it well-behaved through epoch 500.

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

python prepare_resolution_sweep.py

echo "=== LTO_Darcy_bigcap_resaug_normalized, latest (epoch 500) ==="
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_bigcap_resaug_normalized

echo ""
echo "=== LTO_Darcy_bigcap_resaug_normalized, best.pt ==="
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_bigcap_resaug_normalized --lto_epoch best
