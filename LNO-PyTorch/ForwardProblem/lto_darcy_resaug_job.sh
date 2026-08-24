#!/bin/bash
#SBATCH --job-name lto-darcy-resaug
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
# Context-size-augmented variant of LTO_Darcy -- the same fix confirmed
# on the PDEBench Advection track (see
# claude/convcnp-lno-integration-plan.md), now ported to the Darcy/LNO-
# harness side. The original LTO_Darcy run (500 epochs, always encoding
# the full dense 211x211 grid as context) generalizes badly to other
# resolutions (rL2 18.95 at res=85 vs 0.108 at native 211). This run uses
# configs/LTO_Darcy_resaug.jsonc, identical to LTO_Darcy.jsonc except for
# context_frac_min=0.05/context_frac_max=1.0 -- every training batch now
# samples a random context fraction instead of always using the full
# dense grid. The fix lives entirely inside module/convcnp_lto.py's
# ConvCNP_LTO class (training-only, gated on self.training) -- exp.py and
# module/utils.py's shared training loop are unchanged, so this has zero
# effect on LNO_Darcy's own training/checkpoints.
#
# 12h limit, matching LNO_Darcy/LTO_Darcy's own bumped-up time limit
# (both hit the original 4-6h limit before).
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch lto_darcy_resaug_job.sh
# After it finishes, re-run the resolution-transfer eval against it (edit
# evaluate_resolution_transfer.py's exp-name lookup, or add a CLI arg, to
# point at LTO_Darcy_resaug instead of LTO_Darcy) to confirm the fix
# closes the same gap it did for Advection.

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

bash scripts/LTO_Darcy_resaug.sh
