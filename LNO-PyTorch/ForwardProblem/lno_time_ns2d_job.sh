#!/bin/bash
#SBATCH --job-name lno-time-ns2d
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
# NS2d matched-harness LNO retrain (2026-09-21) -- LNO trained at its OWN
# natural/published architecture (n_block=8, n_dim=256 -> 5,076,481 params,
# same as configs/LNO_Darcy.jsonc's convention), on our own hardware/harness,
# rather than only citing LNO's published literature number (0.0845 rL2).
# Same pattern as Darcy's LNO retrain (see lto_darcy_job.sh's sibling job,
# whichever wrapper submitted scripts/LNO_Darcy.sh) -- this is the NS2d
# analogue. See item 4 of Dee's Aug/Sep 2026 Burgers/NS2d punch list and
# claude/convcnp-lno-integration-plan.md's "NS2d" section for the full
# design rationale.
#
# This is the NATURAL-SIZE variant (Darcy-consistent: LNO unshrunk, so the
# "LTO matches/beats it with far fewer params" framing stays intact). A
# separate PARAM-MATCHED variant (~1.1M params, matching LTO's NS2d budget)
# is lno_time_ns2d_small_job.sh -- Dee asked for BOTH framings.
#
# scripts/LNO_NS2d.sh's --device flag was fixed from "3" to "0" before this
# job script was written -- Palmetto's H200 allocation only exposes device
# index 0 for a single-GPU job (same bug LNO_Darcy.sh had, already fixed
# there).
#
# BEFORE SUBMITTING: prepare.py --data_name NS2d (called automatically
# below) needs the raw file NavierStokes_V1e-5_N1200_T20.mat present in
# DATA_PATH -- same requirement as lto_time_ns2d_job.sh, which already
# depends on the same data. If that job has already run successfully,
# prepare.py's cached .npy files may already exist and this step is fast;
# otherwise it re-derives them from the raw .mat file.
#
# Own master_port (12343, set inside scripts/LNO_NS2d.sh) -- doesn't
# collide with LNO_Darcy's (12341) or LTO_time_NS2d's (12350, per
# lto_time_ns2d_job.sh's own comment) or LNO_NS2d_small's (12344, set
# inside scripts/LNO_NS2d_small.sh) -- safe to run all of these
# concurrently.
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch lno_time_ns2d_job.sh
# After it finishes, use the adapted evaluate_ns2d.py (item 2f of the
# punch list) to compare LTO against this checkpoint directly, not just
# LNO's literature number.

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

bash scripts/LNO_NS2d.sh
