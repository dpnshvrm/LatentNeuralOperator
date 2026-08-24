#!/bin/bash
#SBATCH --job-name check-early-checkpoints
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
# Checks whether LTO_Darcy_resaug_normalized_dilated and
# LTO_Darcy_bigcap_resaug were both read at an overfit epoch-500
# checkpoint rather than their true best. dilated's training log showed
# a textbook overfitting curve -- train loss falls smoothly the whole way
# (0.97 -> 0.096) while val loss plateaus/oscillates from ~epoch 60
# onward (best seen ~0.194 around epoch 60-70, vs. 0.212 at epoch 500).
# bigcap_resaug used the same epoch budget and a similarly large capacity
# increase, so it may have the same problem -- never checked before now,
# since every eval so far only ever looked at the final/latest checkpoint.
#
# Pure evaluation against checkpoints that already exist on disk (saved
# every model_save_interval_epoch=50) -- no retraining. All four checks
# run in one job/one GPU allocation instead of four separate submissions.
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch check_early_checkpoints_job.sh

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

python prepare_resolution_sweep.py

echo "=== LTO_Darcy_resaug_normalized_dilated, epoch 50 ==="
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_resaug_normalized_dilated --lto_epoch 50

echo ""
echo "=== LTO_Darcy_resaug_normalized_dilated, epoch 100 ==="
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_resaug_normalized_dilated --lto_epoch 100

echo ""
echo "=== LTO_Darcy_bigcap_resaug, epoch 50 ==="
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_bigcap_resaug --lto_epoch 50

echo ""
echo "=== LTO_Darcy_bigcap_resaug, epoch 100 ==="
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_bigcap_resaug --lto_epoch 100
