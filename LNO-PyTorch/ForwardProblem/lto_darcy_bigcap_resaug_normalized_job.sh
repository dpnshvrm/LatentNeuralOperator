#!/bin/bash
#SBATCH --job-name lto-darcy-bigcap-resaug-normalized
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
# Retest of capacity scaling on Darcy, this time (a) stacked with
# channel_mode="normalized" (LTO_Darcy_bigcap_resaug never got this --
# an oversight caught 2026-08-24) and (b) properly tuned: lower LR (3e-4
# vs 1e-3), higher weight decay (5e-4 vs 5e-5), and best-val-checkpoint
# tracking (module/utils.py's Checkpoint.save_best(), also added this
# batch) instead of relying on whichever numbered epoch landed last.
#
# Purpose: LTO_Darcy_bigcap_resaug (width) and
# LTO_Darcy_resaug_normalized_dilated (receptive field) both landed WORSE
# than the much smaller LTO_Darcy_resaug_normalized (29,139 params) even
# after checking early checkpoints for an overfitting artifact -- see
# claude/convcnp-lno-integration-plan.md's 2026-08-24 decision. But
# neither larger run had its training hyperparameters retuned for the
# larger model, which is a common reason scaling width/depth alone makes
# things WORSE, not better (exactly the pattern observed). This run
# isolates that: if it STILL can't beat resaug_normalized despite being
# properly tuned, that's real evidence of a genuine architecture ceiling,
# not just an artifact of lazy hyperparameters -- answers the "will we
# hit this same wall on other experiments" question either way.
#
# 255,603 params (unchanged from LTO_Darcy_bigcap_resaug -- same widths,
# no dilation added here, to keep this a single-variable-plus-training-fix
# test and stay within the ~3x-fewer-than-LNO efficiency budget already
# used for bigcap_resaug; a dilated+bigcap combo would run ~587,763 params,
# ~77% of LNO's count, which is worth trying only if this run's plain
# combo shows real promise).
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch lto_darcy_bigcap_resaug_normalized_job.sh
# After it finishes, evaluate resolution transfer against both its latest
# epoch and its tracked best checkpoint:
#   sbatch eval_resolution_bigcap_resaug_normalized_job.sh

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

bash scripts/LTO_Darcy_bigcap_resaug_normalized.sh
