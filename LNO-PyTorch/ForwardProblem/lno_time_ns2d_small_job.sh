#!/bin/bash
#SBATCH --job-name lno-time-ns2d-small
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
# NS2d matched-harness LNO retrain, PARAM-MATCHED VARIANT (2026-09-21) --
# LNO shrunk to n_block=4/n_mode=64/n_dim=160/n_head=8 (1,148,385 params,
# +2.0% over LTO's NS2d ae-reconloss checkpoint's 1,126,211 params -- the
# closest grid point found by find_lno_param_config.py's search over
# n_block in {2,3,4,6} x n_dim in {32..160} x n_head in {4,8} x n_mode in
# {64,128,256}, all instantiated from the real module.model.LNO class and
# counted directly, not hand-derived).
#
# THIS IS A DIFFERENT KIND OF EXPERIMENT THAN lno_time_ns2d_job.sh (the
# natural-size ~5.08M-param variant, Darcy-consistent): shrinking LNO's
# architecture changes what the comparison is testing (equal-budget
# accuracy, not "LTO wins with far fewer params"), and has no Darcy-table
# precedent -- Dee explicitly asked for BOTH framings for NS2d, so this
# variant exists alongside the natural-size one, not instead of it.
#
# BEFORE SUBMITTING: same NS2d raw-data prerequisite as
# lno_time_ns2d_job.sh (NavierStokes_V1e-5_N1200_T20.mat in DATA_PATH) --
# prepare.py --data_name NS2d is idempotent, so running this after (or
# concurrently with) the natural-size job is fine.
#
# Own master_port (12344, set inside scripts/LNO_NS2d_small.sh) -- doesn't
# collide with LNO_NS2d's (12343), LNO_Darcy's (12341), or LTO_time_NS2d's
# (12350).
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch lno_time_ns2d_small_job.sh

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

bash scripts/LNO_NS2d_small.sh
