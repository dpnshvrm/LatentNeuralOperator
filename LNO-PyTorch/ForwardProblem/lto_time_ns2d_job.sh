#!/bin/bash
#SBATCH --job-name lto-time-ns2d
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
# NS2d time-evolution experiment (2026-08-27) -- the "true LTO" track: a
# genuine autoregressive time-stepping benchmark, on the SAME data/task
# LNO's own paper reports a number for (Table 1, assets/Forward-1.png:
# NS2d rL2 = 8.45e-2), so this is a direct literature comparison -- LNO
# is NOT retrained here, only ConvCNP_LTO. See the "NS2d" section of
# claude/convcnp-lno-integration-plan.md for the full design rationale
# (why CNNLatentOperator not CNNLatentFlow, why channel_mode=normalized,
# why no context-size augmentation for this run).
#
# BEFORE SUBMITTING: prepare.py --data_name NS2d (called automatically
# below) needs the raw file NavierStokes_V1e-5_N1200_T20.mat present in
# DATA_PATH -- check it's actually there (it's in the same shared Google
# Drive "PDE_datasets" folder the Darcy raw data came from, see this
# repo's README) before running, or prepare.py will fail with a
# FileNotFoundError partway through data prep.
#
# Own master_port (12350) -- doesn't collide with any Darcy/Advection
# job's port (12341-12349), so this is safe to submit and run
# concurrently with anything else already running.
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch lto_time_ns2d_job.sh
# After it finishes, verify the reported number independently (don't
# trust the training log's "Val Loss Full" at face value -- see the
# training-log-vs-eval-script caution in claude/convcnp-lno-integration-plan.md):
#   python evaluate_ns2d.py --lto_exp LTO_time_NS2d --lto_epoch best

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

bash scripts/LTO_time_NS2d.sh
