#!/bin/bash
#SBATCH --job-name lto-darcy-resaug-normalized
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
# channel_mode="normalized" ablation on top of LTO_Darcy_resaug -- the
# still-open Corollary A.2 question (raw [density, signal] vs. normalized
# [density, signal/density] SetConv encoder channel), now checked on Darcy
# specifically. Advection already ran this ablation and found channel_mode
# made ~no difference (0.500 vs 0.484 rL2, within noise) -- this run checks
# whether that holds for Darcy too, where the learned length scale ended up
# much wider (encoder ell ~0.84, near-global -- see the length-scale
# diagnostic in claude/convcnp-lno-integration-plan.md) than on Advection,
# so an unnormalized sum over that near-global kernel is closer to the
# actual mechanism blamed for the original resolution-transfer collapse.
#
# Uses configs/LTO_Darcy_resaug_normalized.jsonc -- identical to
# LTO_Darcy_resaug.jsonc except channel_mode, so this isolates that one
# variable. Own experiment/checkpoint dir (LTO_Darcy_resaug_normalized),
# own master_port (12345) -- doesn't collide with any of the other four
# Darcy jobs (LNO_Darcy 12341, LTO_Darcy 12342, LTO_Darcy_resaug 12343,
# LTO_Darcy_bigcap_resaug 12344), so this is safe to submit and run
# concurrently with lto_darcy_bigcap_resaug_job.sh.
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch lto_darcy_resaug_normalized_job.sh
# After it finishes, evaluate resolution transfer against it:
#   python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_resaug_normalized

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

bash scripts/LTO_Darcy_resaug_normalized.sh
