#!/bin/bash
#SBATCH --job-name eval-resolution-transfer-bigcap-resaug
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
# Same resolution-transfer eval, pointed at LTO_Darcy_bigcap_resaug (the
# capacity-increased variant) instead of LTO_Darcy_resaug. Run this once
# lto_darcy_bigcap_resaug_job.sh finishes.

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

python prepare_resolution_sweep.py
python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_bigcap_resaug
