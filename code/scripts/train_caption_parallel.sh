#!/bin/bash

data_path="{DATA_PATH}"                                                                             # [need to replace]
output_path="{OUTPUT_PATH}"                                                                         # [need to replace]
pretrained_model_path="{HOME}/omni-perception-pretrainer/pretrained_model"                          # [need to replace]
model_config_path="{HOME}/omni-perception-pretrainer/code/model_configs/model_config_finetune.yaml" # [need to replace]
boot_file_path="{HOME}/omni-perception-pretrainer/code/src/scripts/train_caption.py"                # [need to replace]
task_name="finetune_caption"

if [ $# != 3 ]
then
    echo "Usage:
          bash scripts/train_caption_parallel.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [RANK_TABLE_FILE]"
    exit 1
fi

if [ $1 -lt 1 ] || [ $1 -gt 8 ]
then
    echo "error: DEVICE_NUM=$1 is not in [1,8]"
    exit 1
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

rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}
export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))
for((i=0; i<${RANK_SIZE}; i++))
do
    export RANK_ID=$((rank_start + i))
    export DEVICE_ID=${CANDIDATE_DEVICE[i]}
    mkdir -p ${output_path:?}/${task_name:?}/rank_$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    nohup tk finetune \
    --quiet \
    --boot_file_path $boot_file_path \
    --data_path $data_path \
    --output_path $output_path \
    --model_config_path $model_config_path \
    --pretrained_model_path $pretrained_model_path &
done
