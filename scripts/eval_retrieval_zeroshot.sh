#!/bin/bash

export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3

output_dir="output/retrieval"

device_id=7
task_name="zeroshot"
task_config_file="ft_ret_1b.json"
pretrain_ckpt_file="OPT_1-38_136.ckpt"
export RANK_SIZE=1;export DEVICE_ID=$device_id;export MS_COMPILER_CACHE_PATH=${output_dir:?}/${task_name:?}; \
nohup python -u src/scripts/eval_retrieval.py \
    --config=config/retrieval/$task_config_file \
    --output_dir=$output_dir/$task_name \
    --pretrain_ckpt_file=model/retrieval/$pretrain_ckpt_file \
    --use_parallel=False \
    > $output_dir/$task_name/log_eval_zero_shot 2>&1 &