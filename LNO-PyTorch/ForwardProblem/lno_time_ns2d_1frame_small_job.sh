#!/bin/bash
#SBATCH --job-name lno-time-ns2d-1frame-small
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
# 1-FRAME-CONTEXT ABLATION of lno_time_ns2d_small_job.sh, requested
# 2026-09-24: does LNO's advantage over LTO on NS2d shrink or vanish once
# it's given the same single-state input LTO gets, instead of a 10-frame
# history window? Param-matched to the SAME target as the small variant
# (LTO's NS2d ae-reconloss checkpoint, 1,126,211 params) -- re-run via
# find_lno_param_config.py --y1_dim 3 (x_dim=2 + 1 context frame, vs. the
# original small config's --y1_dim 12 = x_dim=2 + 10 frames), since
# shrinking the context window changes branch_projector's input width and
# therefore the parameter count at any given n_block/n_dim/n_head/n_mode.
# Closest grid point found: n_block=4/n_dim=160/n_head=8/n_mode=64 ->
# 1,146,945 params (+1.8% over target; the original small config landed at
# +2.0%, comparably tight). See LNO_time_NS2d_1frame_small.jsonc's own
# comment header for the full derivation.
#
# DATA: trains on NS2d_1frame_{train,val}.npy, produced by
# prepare_ns2d_1frame.py (re-slices the ALREADY-PREPARED NS2d_{train,val}.npy
# down to context = frame 9 only, target = frame 10 only -- the single
# frame immediately preceding the existing Table 2 "Direct" target, so this
# isolates ONLY the context-window-length variable, not also trajectory
# position -- see that script's own docstring for the full reasoning). Does
# NOT touch or overwrite the original NS2d_{train,val}.npy -- every other
# NS2d run in this project keeps reading those unchanged.
#
# BEFORE SUBMITTING: same NS2d raw-data prerequisite as
# lno_time_ns2d_small_job.sh (NavierStokes_V1e-5_N1200_T20.mat in
# DATA_PATH) -- prepare.py --data_name NS2d is idempotent and re-run here
# for safety, then prepare_ns2d_1frame.py re-slices its output (also
# idempotent -- re-running just overwrites NS2d_1frame_{train,val}.npy with
# the same content).
#
# exp.py's train_time/val_time were fixed 2026-09-24 (T = y2.shape[-1]
# instead of a hardcoded T = 10) specifically so this 1-target-frame
# dataset doesn't index past the end of y2 or silently train on
# padded/repeated targets -- a no-op for every other NS2d/Darcy/etc. config
# in this project, since their y2 is already 10 frames wide.
#
# Own master_port (12355, set inside scripts/LNO_NS2d_1frame_small.sh) --
# doesn't collide with LNO_Darcy's (12341), LNO_NS2d's (12343),
# LNO_NS2d_small's (12344), LNO_Darcy_matched's (12353), or any LTO_Darcy/
# LTO_time_NS2d master_port (12345-12351).
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch lno_time_ns2d_1frame_small_job.sh

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /home/dverma/lno-conda

bash scripts/LNO_NS2d_1frame_small.sh
