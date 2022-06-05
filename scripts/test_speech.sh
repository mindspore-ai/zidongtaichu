#!/bin/bash

export GLOG_v=2;export RANK_SIZE=1;export DEVICE_ID=1 && \
python src/scripts/test_speech.py \
--config config/speech/speech.json \
--use_parallel False \
--ckpt_file OPT_audio_3-404000_2.ckpt
