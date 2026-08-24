#!/bin/bash
#SBATCH --job-name lto-darcy-bigcap-resaug
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
# Capacity increase on top of LTO_Darcy_resaug, to try closing some of the
# accuracy gap to LNO (762,113 params) while keeping the resolution-
# invariance fix -- same idea as PDEBench Advection's "combo" run (see
# claude/convcnp-lno-integration-plan.md). Uses configs/LTO_Darcy_bigcap_resaug.jsonc:
# 3x each channel width vs LTO_Darcy_resaug (hidden_channels 16->48,
# latent_channels 8->24, flow_hidden 32->96, decoder_hidden 192) ->
# 255,603 params (confirmed locally before shipping) -- about 8.8x
# LTO_Darcy_resaug's capacity, while staying ~3x FEWER params than LNO
# (deliberately not matching Advection combo's 4x ratio, which would land
# at ~60% of LNO's param count and weaken the efficiency framing more than
# seems worth it). context_frac_min/max unchanged from LTO_Darcy_resaug,
# so capacity is the only changed variable -- isolates its effect, same
# logic as the Advection bigcap/boundeddt/combo ablation.
#
# grid_size is UNCHANGED (64) -- the O(grid_size^2 * n_points) pairwise
# SetConv tensor (the dominant memory cost, ~6GB, see
# module/convcnp_lto.py's memory note) is unaffected by this change; only
# the CNN/MLP activation memory grows, and only linearly with channel
# width, so this should still comfortably fit an H200. Own experiment/
# checkpoint dir (LTO_Darcy_bigcap_resaug), own master_port (12344) --
# doesn't collide with LNO_Darcy/LTO_Darcy/LTO_Darcy_resaug if run in
# parallel with any of them.
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch lto_darcy_bigcap_resaug_job.sh
# After it finishes, evaluate resolution transfer against it:
#   python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_bigcap_resaug

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

bash scripts/LTO_Darcy_bigcap_resaug.sh
