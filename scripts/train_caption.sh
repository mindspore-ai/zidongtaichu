#!/bin/bash

export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3

output_dir="output/caption"

device_id=0
task_name="finetune_caption"
task_config_file="ft_cap_base.json"
rm -rf ${output_dir:?}/${task_name:?}
mkdir -p ${output_dir:?}/${task_name:?}
export RANK_SIZE=1;export DEVICE_ID=$device_id;export MS_COMPILER_CACHE_PATH=${output_dir:?}/${task_name:?}; \
nohup python -u src/scripts/train_caption.py \
    --config=config/caption/$task_config_file \
    --output_dir=$output_dir/$task_name \
    --pretrain_ckpt_file=model/caption/OPT_1-38_136.ckpt \
    --use_parallel=False \
    --data_url="a" --train_url="a" \
    > $output_dir/$task_name/log_train 2>&1 &