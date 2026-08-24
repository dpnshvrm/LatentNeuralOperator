#!/bin/bash
#SBATCH --job-name lto-darcy-resaug-normalized-dilated
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
# Dilated-conv receptive-field fix on top of LTO_Darcy_resaug_normalized
# (our current best/primary Darcy checkpoint). Diagnosed bottleneck: the
# original CNNLatentOperator is 3 plain 3x3 conv layers -- a 7x7 receptive
# field on the 64x64 latent grid, REGARDLESS of channel width. Darcy is a
# steady elliptic PDE (globally-supported solution operator), which a
# 7x7-local CNN cannot represent no matter how wide -- the likely reason
# LTO_Darcy_bigcap_resaug's channel-width increase made accuracy WORSE,
# not better (more numbers in the same too-small window, more overfitting
# risk, no additional reach). This run replaces that 3-layer stack with a
# dilated one (dilations 2,4,8,16,32) whose receptive field covers the
# FULL 64x64 grid -- confirmed locally via a perturbation test (plain:
# 5x5 spread, dilated: full 64x64 spread) before shipping. Adds ~37K
# params (66,131 total vs. resaug_normalized's 29,139) -- still ~11.5x
# fewer than LNO's 762,113, well below bigcap_resaug's failed 255,603.
#
# Uses configs/LTO_Darcy_resaug_normalized_dilated.jsonc (identical to
# LTO_Darcy_resaug_normalized.jsonc plus "use_dilated": true).
# module/convcnp_lto.py's CNNLatentOperator and ConvCNP_LTO, and
# module/utils.py's dispatch line, were both updated to support
# use_dilated (default False, so every existing config/checkpoint is
# completely unaffected -- confirmed backward-compatible state_dict keys
# locally before shipping). Own experiment/checkpoint dir
# (LTO_Darcy_resaug_normalized_dilated), own master_port (12346) -- no
# collision with any of the other five Darcy jobs (LNO_Darcy 12341,
# LTO_Darcy 12342, LTO_Darcy_resaug 12343, LTO_Darcy_bigcap_resaug 12344,
# LTO_Darcy_resaug_normalized 12345).
#
# Submit from LNO-PyTorch/ForwardProblem/:
#   sbatch lto_darcy_resaug_normalized_dilated_job.sh
# After it finishes, evaluate resolution transfer against it:
#   python evaluate_resolution_transfer.py --lto_exp LTO_Darcy_resaug_normalized_dilated

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source /etc/profile
module load anaconda3/2023.09-0
source activate /scratch/dverma/lno-conda

bash scripts/LTO_Darcy_resaug_normalized_dilated.sh
