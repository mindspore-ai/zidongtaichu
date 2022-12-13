#!/bin/bash

export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3

output_dir="output/caption"
beam_width=4
inference_dir="image"
inference_list="dataset/caption/val/val_coco_v2.json"
inference_list=""

device_id=7
task_name="finetune_caption"
task_config_file="ft_cap_base.json"
ckpt_name="10_7076"
ckpt_file="OPT_cap-$ckpt_name.ckpt"
export RANK_SIZE=1;export DEVICE_ID=$device_id;export MS_COMPILER_CACHE_PATH=${output_dir:?}/${task_name:?}; \
nohup python -u src/scripts/inference_caption.py \
                --inference_dir=$inference_dir \
                --inference_list=$inference_list \
                --ckpt_file=$output_dir/$task_name/ckpt/rank_0/$ckpt_file \
                --beam_width=$beam_width \
                > $output_dir/$task_name/log_inference_${ckpt_name}_bw${beam_width} 2>&1 &

