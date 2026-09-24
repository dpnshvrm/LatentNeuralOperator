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
# (29,139 plain-kernel / 240,594 attn+bigcap "ours"), so the headline
# Darcy table compares LTO against an LNO with ~3-26x more parameters.
# This variant shrinks LNO to match one of those, for an equal-capacity
# comparison alongside the existing natural-size one (LNO_Darcy.jsonc is
# kept, not replaced -- same "both framings" choice made for NS2d).
#
# *** BEFORE SUBMITTING ***: configs/LNO_Darcy_matched.jsonc's
# n_block/n_mode/n_dim/n_head are still PLACEHOLDERS (copied from
# LNO_Darcy.jsonc, i.e. NOT yet param-matched to anything). Run
# find_lno_param_config_darcy.py first to find the actual closest grid
# point for your chosen target, e.g.:
#   python find_lno_param_config_darcy.py --target 240594   # match LTO's attn+bigcap "ours"
#   python find_lno_param_config_darcy.py --target 29139    # match LTO's plain-kernel baseline
# then edit configs/LNO_Darcy_matched.jsonc's model block with the
# result before running this job -- otherwise this just retrains
# LNO_Darcy's native size again under a different experiment name.
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
# Submit from LNO-PyTorch/ForwardProblem/, after editing the config:
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
