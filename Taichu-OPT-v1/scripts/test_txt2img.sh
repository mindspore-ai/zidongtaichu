#!/bin/bash

#================inference [conda env: cjpy37]=======================
export RANK_SIZE=1;export DEVICE_ID=7;python src/scripts/test_txt2img.py \
--config=./config/t2i/ftt2i_base.json \
--output_dir=experiments/test_t2i \
--ckpt_file=experiments/test_t2i/ckpt/rank_0/OPT_txt2img-265_1000.ckpt \
--vae_config=src/vqvae_mindspore/config/VQVAEwBN_MSCOCO.yaml \
--vae_ckpt=experiments/VQVAEwBN_MSCOCO/Model_epoch90_step89999.ckpt \
--use_parallel=False --data_url="a" --train_url="a" --inf_txt_file=datasets/txt2img/mscoco/cocodata_zh/reference.txt