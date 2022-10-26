#!/bin/bash

export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3

output_dir="output/caption"

device_id=3
task_name="finetune_caption_standalone"
task_config_file="ft_cap_1b.json"
ckpt_name="10_14153"
ckpt_file="OPT_cap-$ckpt_name.ckpt"
export RANK_SIZE=1;export DEVICE_ID=$device_id;export MS_COMPILER_CACHE_PATH=${output_dir:?}/${task_name:?}; \
nohup python -u src/scripts/eval_caption.py \
    --config=config/caption/$task_config_file \
    --output_dir=$output_dir/$task_name \
    --finetune_ckpt_file=$output_dir/$task_name/ckpt/rank_0/$ckpt_file \
    --use_parallel=False \
    > $output_dir/$task_name/log_eval_10_7076 2>&1 &
