config_name="LTO_time_NS2d"
exp_name="LTO_time_NS2d"
python prepare.py --data_name NS2d
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 12350 \
exp.py \
--config $config_name \
--device "0" \
--exp $exp_name \
--seed 0
