config_name="LTO_Darcy_resaug_normalized_aereconloss_extrap_fullnorm"
exp_name="LTO_Darcy_resaug_normalized_aereconloss_extrap_fullnorm"
python prepare.py --data_name Darcy
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 12352 \
exp.py \
--config $config_name \
--device "0" \
--exp $exp_name \
--seed 0
