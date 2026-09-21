config_name="LNO_time_NS2d_small"
exp_name="LNO_NS2d_small"
python prepare.py --data_name NS2d
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 12344 \
exp.py \
--config $config_name \
--device "0" \
--exp $exp_name \
--seed 0
