#!/bin/bash

if [ $# != 5 ]
then
    echo "Usage:
          bash scripts/train_txt2img_stagetwo.sh [DEVICE_NUM] [VISIABLE_DEVICES(0,1,2,3,4,5,6,7)] [RANK_TABLE_FILE] [CONFIG_FILE] [OUTPUT_PATH]"
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

export RANK_TABLE_FILE=$3
if [ ! -f $3 ]
then
    echo "error: RANK_TABLE_FILE=$3 is not a file"
exit 1
fi

export CONFIG_FILE=$4
if [ ! -f $4 ]
then
    echo "error: CONFIG_FILE=$4 is not a file"
exit 1
fi

# Parallelize training Transformer Decoder
OUTPUT_PATH=$5
i=1
while(($i<$1))
do
    export RANK_ID=$i
    export DEVICE_ID=${CANDIDATE_DEVICE[i]}
    echo "start training for rank ${RANK_ID}, device ${DEVICE_ID}"
    rm -rf ./train_t2i_parallel$i
    mkdir ./train_t2i_parallel$i
    cp -r ./src ./train_t2i_parallel$i
    cp -r ./config/t2i ./train_t2i_parallel$i
    cd ./train_t2i_parallel$i
    env > env.log
    python src/scripts/train_txt2img.py --config=$CONFIG_FILE --output_dir=$OUTPUT_PATH --data_url="" --train_url=""  > log${RANK_ID}.txt 2>&1 &
    cd ..
    let "i++"
done

i=0
export RANK_ID=$i
export DEVICE_ID=${CANDIDATE_DEVICE[i]}
echo "start training for rank ${RANK_ID}, device ${DEVICE_ID}"
python src/scripts/train_txt2img.py --config=$CONFIG_FILE --output_dir=$OUTPUT_PATH --data_url="" --train_url=""
