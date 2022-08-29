#!/bin/bash

export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3

output_dir="output/vqa"

device_id=0
task_name="finetune_vqa"
task_config_file="ft_vqa_base.json"
rm -rf ${output_dir:?}/${task_name:?}
mkdir -p ${output_dir:?}/${task_name:?}
export RANK_SIZE=1;export DEVICE_ID=$device_id;export MS_COMPILER_CACHE_PATH=${output_dir:?}/${task_name:?}; \
nohup python -u src/scripts/train_vqa.py \
    --config=config/vqa/$task_config_file \
    --output_dir=$output_dir/$task_name \
    --ckpt_file=model/vqa/OPT_1-38_136.ckpt \
    --use_parallel=False \
    --data_url="a" --train_url="a" --mode="train" \
    > $output_dir/$task_name/log_train 2>&1 &