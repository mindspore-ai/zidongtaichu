#!/bin/bash

output_dir="output/vqa"
task_name="finetune_vqa"
task_config_file="ft_vqa_base.json"

if [ $# != 3 ]
then
    echo "Usage:
          bash scripts/train_vqa_parallel.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [RANK_TABLE_FILE]"
exit 1
fi

if [ $1 -lt 1 ] || [ $1 -gt 8 ]
then
    echo "error: DEVICE_NUM=$1 is not in [1,8]"
fi

VISIABLE_DEVICES=$2
IFS="," read -r -a CANDIDATE_DEVICE <<< "$VISIABLE_DEVICES"
if [ ${#CANDIDATE_DEVICE[@]} -ne $1 ]
then
    echo "error: DEVICE_NUM=$1 is not matched with VISIABLE_DEVICES=$2"
    exit 1
fi

if [ ! -f $3 ]
then
    echo "error: RANK_TABLE_FILE=$3 is not a file"
exit 1
fi

export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export HCCL_CONNECT_TIMEOUT=600

ulimit -u unlimited
ulimit -SHn 65535
export DEVICE_NUM=$1
export RANK_SIZE=$1
RANK_TABLE_FILE=$(realpath $3)
export RANK_TABLE_FILE
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

rm -rf ${output_dir:?}/${task_name:?}
mkdir -p ${output_dir:?}/${task_name:?}
export MS_COMPILER_CACHE_PATH=${output_dir:?}/${task_name:?}
export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))
for((i=0; i<${RANK_SIZE}; i++))
do
    export RANK_ID=$((rank_start + i))
    export DEVICE_ID=${CANDIDATE_DEVICE[i]}
    mkdir -p ${output_dir:?}/${task_name:?}/rank_$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > $output_dir/$task_name/rank_$i/env.log
    nohup python -u src/scripts/train_vqa.py \
        --config=config/vqa/$task_config_file \
        --output_dir=$output_dir/$task_name \
        --ckpt_file=model/vqa/OPT_1-38_136.ckpt \
        --use_parallel=True \
        --data_url="a" --train_url="a" --mode="train" \
        > $output_dir/$task_name/rank_$i/log_train 2>&1 &
done
