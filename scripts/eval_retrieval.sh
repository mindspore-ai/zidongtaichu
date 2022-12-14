#!/bin/bash

export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3

output_dir="output/retrieval"

device_id=7
task_name="finetune_retrieval"
task_config_file="ft_ret_1b.json"
ckpt_name="3_25635"
ckpt_file="OPT_ret-$ckpt_name.ckpt"
export RANK_SIZE=1;export DEVICE_ID=$device_id;export MS_COMPILER_CACHE_PATH=${output_dir:?}/${task_name:?}; \
nohup python -u src/scripts/eval_retrieval.py \
    --config=config/retrieval/$task_config_file \
    --output_dir=$output_dir/$task_name \
    --finetune_ckpt_file=$output_dir/$task_name/ckpt/rank_0/$ckpt_file \
    --use_parallel=False \
    > $output_dir/$task_name/log_eval_$ckpt_name 2>&1 &