#!/bin/bash
#SBATCH --job-name lto-darcy-resaug-normalized-attn
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
# PhCA-style learned attention encoder (use_attention_encoder=True),
# replacing the fixed Gaussian SetConv kernel, on top of our LOCKED FINAL
# Darcy baseline LTO_Darcy_resaug_normalized (see claude/convcnp-lno-
# integration-plan.md's 2026-08-25 decision -- resaug_normalized is what
# the paper reports; this is a follow-on architecture ablation, not a
# replacement for it).
#
# Motivation: comparing our SetConv encoder against LNO's PhCA
# (reports/lno_architecture_notes.md) surfaced the most likely remaining
# structural gap -- PhCA's learned attention weighting has NO locality
# bias, while our Gaussian kernel is a fixed, distance-decaying function
# that structurally cannot put high weight on a far context point. This
# is upstream of both prior interventions (use_dilated fixed the
# PROPAGATOR's receptive field; bigcap_resaug{,_normalized} added
# capacity) -- neither touches the encoder's weighting mechanism, which
# is exactly what this run changes.
#
# attn_dim=12 in configs/LTO_Darcy_resaug_normalized_attn.jsonc is chosen
# to param-MATCH resaug_normalized's encoder (10,196 vs. 10,441 params,
# 28,894 vs. 29,139 total) -- this isolates the weighting MECHANISM, not
# a capacity confound like bigcap_resaug_normalized had.
#
# THEORY CAVEAT (also in the config and module/convcnp_lto.py's
# AttentiveSetConvEncoder2D docstring): this is an empirical architecture
# ablation. Theorem 2.3/Corollary 2.4's discretization-rate bound is
# derived for a symmetric, bandwidth-parameterized kernel and does NOT
# automatically transfer to a learned-attention weighting -- do not cite
# this run as validating that theorem. Lemma A.1/Corollary A.2's
# Lipschitz-stability argument plausibly still holds (only needs
# non-negative weights summing to 1) but is unproven for this class.
#
# Own experiment/checkpoint dir (LTO_Darcy_resaug_normalized_attn), own
# master_port (12348) -- doesn't collide with any other Darcy job
# (LNO_Darcy 12341, LTO_Darcy 12342, LTO_Darcy_resaug 12343,
# LTO_Darcy_bigcap_resaug 12344, LTO_Darcy_resaug_normalized 12345,
# LTO_Darcy_resaug_normalized_dilated 12346,
# LTO_Darcy_bigcap_resaug_normalized 12347).
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch lto_darcy_resaug_normalized_attn_job.sh
# After it finishes, evaluate resolution transfer against it:
#   sbatch eval_resolution_resaug_normalized_attn_job.sh
#
# BEFORE SUBMITTING: this run relies on best-checkpoint tracking
# (module/utils.py's Checkpoint.save_best() + the save_best() CALL in
# exp.py's train()). That exp.py patch reverted once already on the
# Windows copy despite a successful-looking push (see
# claude/convcnp-lno-integration-plan.md's "exp.py drift incident" note).
# Verify it's actually present on Palmetto before relying on
# --lto_epoch best:
#   grep -n "save_best\|best_val_loss" exp.py module/utils.py

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

bash scripts/LTO_Darcy_resaug_normalized_attn.sh
