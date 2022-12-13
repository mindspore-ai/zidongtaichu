export RANK_ID=0;export RANK_SIZE=1;export DEVICE_ID=0;\
python vit_eval.py --image_path /store0/images/cc3m/0/training/ \
--npz_save_path /store0/image_feat/cc3m/training/


export RANK_ID=0;export RANK_SIZE=1;export DEVICE_ID=1;\
python vit_eval.py --image_path /store0/images/cc3m/2/training/ \
--npz_save_path /store0/image_feat/cc3m/training/

export RANK_ID=0;export RANK_SIZE=1;export DEVICE_ID=0;\
python vit_eval.py --image_path /store0/images/cc3m/1/training/ \
--npz_save_path /store0/image_feat/cc3m/training/

export RANK_ID=0;export RANK_SIZE=1;export DEVICE_ID=1;\
python vit_eval.py --image_path /store0/images/cc3m/3/training/ \
--npz_save_path /store0/image_feat/cc3m/training/


export RANK_ID=0;export RANK_SIZE=1;export DEVICE_ID=5;\
python vit_eval.py --image_path /store0/images/cc3m/4/training/ \
--npz_save_path /store0/image_feat/cc3m/training/



export RANK_ID=0;export RANK_SIZE=1;export DEVICE_ID=5;\
python vit_eval.py --image_path /store0/images/cc3m/5/training/ \
--npz_save_path /store0/image_feat/cc3m/training/

export RANK_ID=0;export RANK_SIZE=1;export DEVICE_ID=6;\
python vit_eval.py --image_path /store0/images/cc3m/6/training/ \
--npz_save_path /store0/image_feat/cc3m/training/

export RANK_ID=0;export RANK_SIZE=1;export DEVICE_ID=7;\
python vit_eval.py --image_path /store0/images/cc3m/6/validation/ \
--npz_save_path /store0/image_feat/cc3m/validation/

