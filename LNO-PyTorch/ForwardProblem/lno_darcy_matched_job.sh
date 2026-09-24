#!/bin/bash
#SBATCH --job-name lno-darcy-matched
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
# Darcy param-matched LNO variant (2026-09-24) -- same idea as
# lno_time_ns2d_small_job.sh for NS2d: LNO_Darcy.jsonc's native
# architecture (n_block=4/n_mode=256/n_dim=128/n_head=8, 762,113 params)
# is far larger than either of our published LTO Darcy configs
# (29,139 plain-kernel / 28,894 attn-only), so the headline Darcy table
# compares LTO against an LNO with ~26x more parameters. This variant
# shrinks LNO to match those, for an equal-capacity comparison
# alongside the existing natural-size one (LNO_Darcy.jsonc is kept, not
# replaced -- same "both framings" choice made for NS2d).
#
# configs/LNO_Darcy_matched.jsonc's n_block=3/n_mode=128/n_dim=26/n_head=2
# (29,171 params) was found and verified directly by instantiating the
# real module.model.LNO class (not hand-derived) -- see that config's
# own header comment for the exact search command. +0.11% over the
# plain-kernel LTO target, +0.96% over the attn-only target -- one
# config serves both comparisons.
#
# Own master_port (12353, set inside scripts/LNO_Darcy_matched.sh) --
# doesn't collide with any other Darcy/NS2d script (highest previously
# used in this family is 12352, LTO_Darcy_resaug_normalized_aereconloss_extrap_fullnorm).
#
# Uses /home/dverma/lno-conda (not /scratch/dverma/lno-conda) -- the
# newer env path, per the init_fs_encoding/codec fatal-error fix noted
# in lto_darcy_resaug_normalized_aereconloss_extrap_job.sh's own header;
# this is a new job that hasn't run yet, so it should use the fixed path.
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch lno_darcy_matched_job.sh
# Once it finishes, it can be evaluated the same way the other Darcy
# checkpoints are (evaluate_resolution_transfer.py currently hardcodes
# LNO_Darcy as its LNO baseline -- ask if you want a forked eval script
# that takes the LNO experiment name as a flag, to compare against this
# matched variant instead).

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /home/dverma/lno-conda

bash scripts/LNO_Darcy_matched.sh
