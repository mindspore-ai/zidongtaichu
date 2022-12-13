#!/bin/bash

export GLOG_v=2;export RANK_SIZE=1;export DEVICE_ID=1 && \
python -u src/scripts/train_speech.py \
--config config/speech/speech.json \
--use_parallel False

