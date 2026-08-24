config_name="LTO_Darcy_resaug"
exp_name="LTO_Darcy_resaug"
python prepare.py --data_name Darcy
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 12343 \
exp.py \
--config $config_name \
--device "0" \
--exp $exp_name \
--seed 0
