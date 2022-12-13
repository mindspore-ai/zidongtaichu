
device_id=0

export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export DEVICE_ID=$device_id
export RANK_ID=0
export RANK_SIZE=1
export RANK_TABLE_FILE=hccl_1p_0_192.168.88.192.json

python -u src/scripts/pretrain.py \
        --config config/pretrain/opt_v2_base.json \
        --use_parallel "true"