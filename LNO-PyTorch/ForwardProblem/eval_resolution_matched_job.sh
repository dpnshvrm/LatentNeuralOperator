#!/bin/bash
#SBATCH --job-name eval-resolution-transfer-matched
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
# Evaluates LNO_Darcy_matched (the param-matched LNO Darcy variant,
# 29,171 params -- see configs/LNO_Darcy_matched.jsonc and
# find_lno_param_config_darcy.py) against BOTH published LTO Darcy
# baselines it was matched to: the plain-kernel encoder
# (LTO_Darcy_resaug_normalized, 29,139 params, +0.11% off) and the
# attn-only encoder (LTO_Darcy_resaug_normalized_attn, 28,894 params,
# +0.96% off) -- one matched LNO config serves both comparisons.
#
# Uses evaluate_resolution_transfer_matched.py -- a fork of the golden
# evaluate_resolution_transfer.py (never edited in place) that adds
# --lno_exp/--lno_config/--lno_epoch, mirroring the --lto_exp family the
# golden script already has for the LTO side. Defaults reproduce the
# golden script's original LNO_Darcy vs LTO_Darcy comparison exactly, so
# only this job (and future scripts pointing at other LNO experiments)
# need the fork -- eval_resolution_resaug_normalized_job.sh and friends
# keep using the golden script unmodified.
#
# Run this once lno_darcy_matched_job.sh has finished (check for
# experiments/LNO_Darcy_matched/checkpoint/*.pt).

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
# /home/dverma/lno-conda -- the fixed env path (see lno_darcy_matched_job.sh's
# own header for the /scratch init_fs_encoding bug this avoids).
source activate /home/dverma/lno-conda

python prepare_resolution_sweep.py

echo "=== LNO_Darcy_matched vs LTO_Darcy_resaug_normalized (plain kernel) ==="
python evaluate_resolution_transfer_matched.py --lno_exp LNO_Darcy_matched --lto_exp LTO_Darcy_resaug_normalized

echo ""
echo "=== LNO_Darcy_matched vs LTO_Darcy_resaug_normalized_attn (learned attention) ==="
python evaluate_resolution_transfer_matched.py --lno_exp LNO_Darcy_matched --lto_exp LTO_Darcy_resaug_normalized_attn
