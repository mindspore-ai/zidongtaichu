#!/bin/bash

device_id=0

export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export RANK_SIZE=1
export DEVICE_ID=$device_id

python -u src/scripts/eval_caption.py \
    --config=config/caption/eval_caption_base.json \
    --use_parallel=false