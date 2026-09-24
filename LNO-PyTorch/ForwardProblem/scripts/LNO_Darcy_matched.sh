config_name="LNO_Darcy_matched"
exp_name="LNO_Darcy_matched"
python prepare.py --data_name Darcy
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 12353 \
exp.py \
--config $config_name \
--device "0" \
--exp $exp_name \
--seed 0
