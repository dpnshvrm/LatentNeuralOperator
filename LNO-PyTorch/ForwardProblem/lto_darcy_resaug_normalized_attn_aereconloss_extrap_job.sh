#!/bin/bash
#SBATCH --job-name lto-darcy-resaug-normalized-attn-aereconloss-extrap
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
# Same two changes as lto_darcy_resaug_normalized_aereconloss_extrap_job.sh
# (AE reconstruction loss + context_frac_max 1.0->4.0), applied here on
# top of the learned-attention encoder checkpoint
# (LTO_Darcy_resaug_normalized_attn, 28,894 params) instead of the plain
# kernel. See that job's own comments for the full mechanism description.
#
# Note on context_frac_max here specifically: this checkpoint is NOT
# fixing a known break -- LTO_Darcy_resaug_normalized_attn is already
# flat/best at the super-native eval point (resolution 421: 0.04121, its
# own BEST point on the curve, see claude/convcnp-lno-integration-plan.md's
# 2026-08-26 result). Included anyway for a clean, consistent comparison
# against the plain-kernel variant's own extrap+AE-loss run, per Dee's
# explicit choice (both encoder variants get both changes) -- not because
# the attention encoder needs this fix. The AE loss IS a genuinely new,
# previously-untried lever for this checkpoint though.
#
# Own experiment/checkpoint dir (LTO_Darcy_resaug_normalized_attn_aereconloss_extrap),
# own master_port (12351) -- doesn't collide with any other Darcy job,
# including the sibling job above (12350).
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch lto_darcy_resaug_normalized_attn_aereconloss_extrap_job.sh
# After it finishes, evaluate resolution transfer against it:
#   sbatch eval_resolution_resaug_normalized_attn_aereconloss_extrap_job.sh
#
# BEFORE SUBMITTING: same verification as the sibling job --
#   grep -n "needs_y2\|rL2_ae\|RelLpLossWithRecon\|compute_recon_loss" exp.py module/utils.py module/loss.py module/convcnp_lto.py
#
# CONDA_ENV_PATH (UPDATED 2026-09-23): same fix as the sibling job --
# /scratch/dverma/lno-conda hit the init_fs_encoding fatal error, same
# root cause as the earlier lto-conda /scratch bug. Switched to a fresh
# /home/dverma/lno-conda (Python 3.9, same package list). See the
# sibling job's own comment for the full note.

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /home/dverma/lno-conda

bash scripts/LTO_Darcy_resaug_normalized_attn_aereconloss_extrap.sh
