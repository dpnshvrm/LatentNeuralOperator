#!/bin/bash
#SBATCH --job-name eval-resolution-transfer-resaug
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
# Same resolution-transfer eval as eval_resolution_job.sh, pointed at
# LTO_Darcy_resaug (the context-size-augmentation fix) instead of the
# original LTO_Darcy. Run this once lto_darcy_resaug_job.sh finishes, to
# check whether the fix closes the same resolution-transfer gap on Darcy
# that it did on PDEBench Advection.
#
# darcy_sweep_r{85,106,141,421}.npy already exist from the first run
# (prepare_resolution_sweep.py skips resolutions that already exist), so
# this doesn't need to regenerate them -- but running it again is
# harmless/fast either way.

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

python prepare_resolution_sweep.py
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_resaug
