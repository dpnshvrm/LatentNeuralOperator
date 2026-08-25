#!/bin/bash
#SBATCH --job-name eval-resolution-transfer-bigcap-resaug-normalized-best
#SBATCH --partition=nextlab200
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#
# Standalone re-run of JUST the best.pt eval for LTO_Darcy_bigcap_resaug_normalized
# -- the "latest" (epoch 500) eval already succeeded and printed a full table
# (see eval-resolution-transfer-bigcap-resaug-normalized-<jobid>.out), but the
# best.pt eval's output never appeared: 1h wasn't enough time for BOTH full
# 5-resolution sweeps back to back, so this one was almost certainly still
# running when the job hit its walltime and got killed -- and since Python
# fully buffers stdout when writing to a redirected file (not a live
# terminal), whatever it had already printed was sitting unflushed in memory
# and was lost when the process was killed, rather than partially appearing
# in the log. Two fixes here: (1) 2h budget instead of splitting the work
# with an identical-cost sibling run, (2) `python -u` (unbuffered stdout) so
# any future kill still leaves behind whatever had actually printed so far,
# instead of losing all of it silently.
#
# prepare_resolution_sweep.py isn't re-run here -- the sweep .npy files were
# already confirmed cached ("already exists") in the previous run's log.
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch eval_resolution_bigcap_resaug_normalized_best_job.sh

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

echo "=== LTO_Darcy_bigcap_resaug_normalized, best.pt ==="
python -u evaluate_resolution_transfer.py --lto_exp LTO_Darcy_bigcap_resaug_normalized --lto_epoch best
