config_name="LNO_time_NS2d_1frame_small"
exp_name="LNO_NS2d_1frame_small"
python prepare.py --data_name NS2d
python prepare_ns2d_1frame.py --data_path ./datas
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 12355 \
exp.py \
--config $config_name \
--device "0" \
--exp $exp_name \
--seed 0
