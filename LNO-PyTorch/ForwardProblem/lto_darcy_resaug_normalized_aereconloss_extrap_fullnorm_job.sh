#!/bin/bash
#SBATCH --job-name lto-darcy-resaug-normalized-aereconloss-extrap-fullnorm
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
# Runs IN PARALLEL with lto_darcy_resaug_normalized_aereconloss_extrap_job.sh
# (already submitted/running as of 2026-09-23), not a replacement for it --
# own experiment dir (LTO_Darcy_resaug_normalized_aereconloss_extrap_fullnorm)
# and own master_port. Checked every scripts/*.sh master_port on Palmetto
# directly (not just the locally-staged subset) before picking one: highest
# in use is 12351 (LTO_Darcy_resaug_normalized_attn_aereconloss_extrap.sh),
# so this uses 12352.
#
# Only ONE change vs. that sibling job: channel_mode "normalized" ->
# "fully_normalized" (see configs/LTO_Darcy_resaug_normalized_aereconloss_
# extrap_fullnorm.jsonc's comment and module/convcnp_lto.py's
# SetConvEncoder2D.__init__ for the mechanism). This is the same fix that
# motivated the NS2d root-cause finding: "normalized" mode only ever
# normalizes the encoder's `signal` channel, never `density` itself, which
# stays a raw w.sum() that scales with the context point count. Darcy's
# own worst extrapolation point (resolution 421: 0.218 rL2, 25.6x gap to
# LNO -- see claude/convcnp-lno-integration-plan.md's 2026-08-23 result)
# is the same "super-native context" regime the NS2d diagnostic found the
# encoder broken in, so this checks whether the same fix helps here too --
# NOT yet directly confirmed on Darcy, this run is that confirmation.
#
# Same CONDA_ENV_PATH fix as its sibling job (env rebuilt at
# /home/dverma/lno-conda after the /scratch/dverma/lno-conda
# init_fs_encoding failure).
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch lto_darcy_resaug_normalized_aereconloss_extrap_fullnorm_job.sh
# After it finishes, evaluate resolution transfer the same way as its
# sibling (build an eval job script pointed at this run's exp_name/checkpoint
# the same way eval_resolution_resaug_normalized_aereconloss_extrap_job.sh
# was built for the normalized run).
#
# BEFORE SUBMITTING: verify module/convcnp_lto.py's fully_normalized
# addition actually landed on Palmetto (same "exp.py drift incident" check
# as its sibling job's own header note):
#   grep -n "fully_normalized" module/convcnp_lto.py

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /home/dverma/lno-conda

bash scripts/LTO_Darcy_resaug_normalized_aereconloss_extrap_fullnorm.sh
