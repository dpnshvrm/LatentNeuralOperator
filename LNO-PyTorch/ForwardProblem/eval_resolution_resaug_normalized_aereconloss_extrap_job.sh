#!/bin/bash
#SBATCH --job-name eval-resolution-transfer-resaug-normalized-aereconloss-extrap
#SBATCH --partition=nextlab200
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --time 01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#
# Evaluates LTO_Darcy_resaug_normalized_aereconloss_extrap (plain kernel +
# AE loss + context_frac_max=4.0) at both its latest (epoch 500) checkpoint
# and its tracked best.pt (best.pt here is still keyed on pure forecast
# rL2 -- RelLpLossWithRecon's recon term is always zero in eval mode, see
# module/loss.py's docstring -- so best.pt/latest stay comparable to
# every other Darcy checkpoint's own numbers). No script changes needed --
# evaluate_resolution_transfer.py already takes an arbitrary --lto_exp.
# Run this once lto_darcy_resaug_normalized_aereconloss_extrap_job.sh
# finishes. Compare against LTO_Darcy_resaug_normalized's own numbers
# (claude/convcnp-lno-integration-plan.md) at all 5 resolutions,
# especially resolution 421 (the point this checkpoint's context_frac_max
# extension specifically targets).

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

python prepare_resolution_sweep.py

echo "=== LTO_Darcy_resaug_normalized_aereconloss_extrap, latest (epoch 500) ==="
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_resaug_normalized_aereconloss_extrap

echo ""
echo "=== LTO_Darcy_resaug_normalized_aereconloss_extrap, best.pt ==="
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_resaug_normalized_aereconloss_extrap --lto_epoch best
