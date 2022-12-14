# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import time
import argparse
import json
import math
import numpy as np
from pathlib2 import Path
import mindspore as ms
from mindspore import context, ops
from mindspore.train.model import Model
from mindspore.common.tensor import Tensor
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.train.serialization import load_checkpoint, load_param_into_net

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")))
from src.config.config import *
from src.data import create_dataset, get_batch_data
from src.data.utils import pad_sequence
from src.model_mindspore.parallel_transformer import ParallelConfig
from src.model_mindspore.retrieval_ms import UniterThreeForPretrainingForRetFinetuneEval
from src.tools.misc import parse_with_config, set_random_seed

def init_env(opts):
    """ init_env """
    if opts.use_parallel:
        device_id = int(os.getenv('DEVICE_ID', '0'))
        print('device_id:{}'.format(device_id))
        rank_id_str = os.getenv('RANK_ID', '0')
        rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])
        print('rank_id:{}'.format(rank_id), "rank_id str:{}".format(rank_id_str))
    else:
        device_num = 1
        device_id = int(os.getenv('DEVICE_ID', '0'))
        rank = 0
        rank_id = 0
        opts.rank = rank
    local_rank = rank_id
    print('local_rank:{}, device id:{}'.format(local_rank, device_id))
    profiling_path = os.path.join(opts.output_dir, f'cache/{local_rank}-graphs/')
    if not os.path.exists(profiling_path):
        Path(profiling_path).mkdir(parents=True, exist_ok=True)
    time.sleep(1)
    save_graphs_path = os.path.join(profiling_path, "graph")
    if not os.path.exists(save_graphs_path):
        Path(save_graphs_path).mkdir(parents=True, exist_ok=True)

    strategy_ckpt_save_file = save_graphs_path + \
                              "strategy" + str(local_rank) + ".ckpt"

    os.environ['HCCL_CONNECT_TIMEOUT'] = "6000"
    os.system('ulimit -s 102400')
    set_random_seed(opts.seed)

    print('local_rank:{}, device id:{} start to run...'.format(
        local_rank, device_id), flush=True)

    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        save_graphs_path=save_graphs_path,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(max_device_memory="30GB")
    context.set_context(reserve_class_name_in_scope=False)

    if opts.use_parallel:
        init()
        print("start init")

        device_num = get_group_size()
        rank = get_rank()
        opts.rank = rank
        print("device_id is {}, rank_id is {}, device_num is {}".format(
            device_id, rank, device_num))
        context.reset_auto_parallel_context()
        
        parallel_optimizer_config = {"gradient_accumulation_shard": True}
        optimizer_weight_shard_size = opts.optimizer_shard_size
        print("===========optimizer_weight_shard_size============", optimizer_weight_shard_size)
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
            full_batch=opts.full_batch,
            enable_alltoall=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=opts.enable_parallel_optimizer,
            strategy_ckpt_save_file=strategy_ckpt_save_file,
            parallel_optimizer_config=parallel_optimizer_config,
            optimizer_weight_shard_size=optimizer_weight_shard_size
        )
        
        ParallelConfig.mp = opts.tensor_shard_size
        ParallelConfig.dp = device_num // ParallelConfig.mp
        ParallelConfig.optimizer_shard = opts.enable_parallel_optimizer
        if opts.full_batch:
            opts.val_batch_size = int(opts.val_batch_size * ParallelConfig.dp)
        print((f"=====device_num:{device_num} dp:{ParallelConfig.dp} mp:{ParallelConfig.mp} "
               f"enable_optimizer_shard:{ParallelConfig.optimizer_shard} op:{opts.optimizer_shard_size} " 
               f"val_batch_size:{opts.val_batch_size}"))
    else:
        device_num = 1

    # init dataset
    ds = create_dataset(opts, device_num=device_num, is_train=False)
    dataset_size = ds.get_dataset_size()
    print("=====dataset size: ", dataset_size, flush=True)

    return rank_id, ds

def load_pretrain_ckpt(net, ckpt_file):
    if not ckpt_file:
        return
    print('start loading pretrain ckpt:', ckpt_file)
    param_dict = load_checkpoint(ckpt_file)
    if param_dict:
        param_dict["uniter.img_embeddings.img_linear.weight"] = param_dict["feat_regress.weight"]
        param_dict["uniter.audio_embeddings.audio_linear.weight"] = param_dict["audio_feat_regress.weight"]
        param_dict["uniter.embeddings.word_embeddings.embedding_table"] = param_dict["cls.predictions.decoder.weight"]
        param_dict["rank_output.weight"] = ms.Parameter(param_dict["itm_output.weight"].data[2:3, :])
        param_dict["rank_output.bias"] = ms.Parameter(param_dict["itm_output.bias"].data[2:3])
        param_not_load = load_param_into_net(net, param_dict)
        print("==========param not load==========:", param_not_load)
        print('load pretrain ckpt finished:', ckpt_file)

def load_finetune_ckpt(net, ckpt_file):
    if not ckpt_file:
        return
    print('start loading finetune ckpt:', ckpt_file)
    param_dict = load_checkpoint(ckpt_file)
    if param_dict:
        param_not_load = load_param_into_net(net, param_dict)
        print("==========param not load==========:", param_not_load)
        print('load finetune ckpt finished:', ckpt_file)

def load_distributed_pretrain_ckpt(net, rank_id, ckpt_size, ckpt_path, ckpt_name):
    if not ckpt_path or not ckpt_name:
        return
    ckpt_id = rank_id % ckpt_size
    folder = "rank_" + str(ckpt_id)
    ckpt_file = os.path.join(ckpt_path, folder, ckpt_name)
    if not os.path.exists(ckpt_file):
        print(f"ckpt_file:{ckpt_file} does not exists")
        return
    print('start loading distributed pretrain ckpt:', ckpt_file)  
    param_dict = load_checkpoint(ckpt_file)
    if param_dict:
        param_dict["uniter.img_embeddings.img_linear.weight"] = param_dict["feat_regress.weight"]
        param_dict["uniter.audio_embeddings.audio_linear.weight"] = param_dict["audio_feat_regress.weight"]
        param_dict["uniter.embeddings.word_embeddings.embedding_table"] = param_dict["cls.predictions.decoder.weight"]
        param_dict["rank_output.weight"] = ms.Parameter(param_dict["itm_output.weight"].data[2:3, :])
        param_dict["rank_output.bias"] = ms.Parameter(param_dict["itm_output.bias"].data[2:3])
        param_not_load = load_param_into_net(net, param_dict)
        print("==========param not load==========:", param_not_load)
        print('load distributed pretrain ckpt finished:', ckpt_file)

def load_distributed_finetune_ckpt(net, rank_id, ckpt_size, ckpt_path, ckpt_name):
    if not ckpt_path or not ckpt_name:
        return
    ckpt_id = rank_id % ckpt_size
    folder = "rank_" + str(ckpt_id)
    ckpt_file = os.path.join(ckpt_path, folder, ckpt_name)
    if not os.path.exists(ckpt_file):
        print(f"ckpt_file:{ckpt_file} does not exists")
        return
    print('start loading distributed finetune ckpt:', ckpt_file)
    param_dict = load_checkpoint(ckpt_file)
    if param_dict:        
        param_not_load = load_param_into_net(net, param_dict)
        print("==========param not load==========:", param_not_load)
        print('load distributed finetune ckpt finished:', ckpt_file)

def create_fake_input(batch_size):
    input_ids = Tensor(np.array(np.random.randint(VOCAB_SIZE, size=(batch_size, MAX_FULL_TEXT_LEN)), dtype=np.int32), ms.int32)
    position_ids = Tensor(np.expand_dims(np.arange(0, MAX_FULL_TEXT_LEN, dtype=np.int32), axis=0), ms.int32)
    img_feat = Tensor(np.random.rand(batch_size, MAX_IMG_LEN, PATCH_SIZE ** 2 * 3), ms.float32)
    img_pos_feat = Tensor(np.expand_dims(np.arange(0, MAX_IMG_LEN, dtype=np.int32), axis=0), ms.int32)
    audio_feat = Tensor(np.zeros((batch_size, MAX_AUDIO_LEN, AUDIO_DIM), dtype=np.float32), ms.float32)
    audio_pos_ids = Tensor(np.zeros((1, MAX_AUDIO_LEN), dtype=np.int32), ms.int32)
    attention_mask = Tensor(pad_sequence(np.ones((batch_size, MAX_IMG_TEXT_LEN), dtype=np.int32), batch_first=True, padding_value=0, max_lens=MAX_FULL_LEN), ms.int32)
    gather_index = Tensor(np.repeat(np.expand_dims(np.arange(0, MAX_FULL_LEN, dtype=np.int32), axis=0), batch_size, axis=0), ms.int32)
    img_masks = Tensor(np.zeros((batch_size, MAX_IMG_LEN), dtype=np.bool_), ms.bool_)
    
    return (input_ids, position_ids, img_feat, img_pos_feat, audio_feat, audio_pos_ids,
                  attention_mask, gather_index, img_masks)
    
def main(opts):
    # init
    (rank_id, ds) = init_env(opts)
    # eval
    net_without_loss = UniterThreeForPretrainingForRetFinetuneEval(opts.model_config, img_dim=IMG_DIM,
                                                                    img_label_dim=IMG_LABEL_DIM,
                                                                    audio_dim=AUDIO_DIM, audio_label_dim=AUDIO_LABEL_DIM,
                                                                    use_txt_out=opts.use_txt_out, use_video=opts.use_video,
                                                                    full_batch=opts.full_batch, use_moe=opts.use_moe,
                                                                    args=opts, is_parallel=opts.use_parallel)

    model = Model(net_without_loss)

    (input_ids, position_ids, img_feat, img_pos_feat, audio_feat, audio_pos_ids, attention_mask, gather_index, img_masks) = \
    create_fake_input(opts.val_batch_size)
    model.predict(input_ids, position_ids, img_feat, img_pos_feat, audio_feat, audio_pos_ids, attention_mask, gather_index, img_masks) 
    
    # load ckpt
    load_pretrain_ckpt(net_without_loss, opts.pretrain_ckpt_file)
    load_finetune_ckpt(net_without_loss, opts.finetune_ckpt_file)
    load_distributed_pretrain_ckpt(net_without_loss, rank_id, opts.ckpt_size, opts.ckpt_path, opts.pretrain_ckpt_name)
    load_distributed_finetune_ckpt(net_without_loss, rank_id, opts.ckpt_size, opts.ckpt_path, opts.finetune_ckpt_name)
    
    ids = json.load(open(opts.ids_val_path,'r'))
    print("retrieval dataset's length is: ", len(ids))
    log = validate_itm_matching(model, ds, len(ids), is_parallel=opts.use_parallel)
    print(log)

def validate_itm_matching(model, val_ds, pair_num=1000, is_parallel = True):
    topk = ops.TopK()
    print("start running ITM validation...")
    score_vec = Tensor(np.zeros((pair_num ** 2,)), ms.float32)
    n_ex = 0
    for batch in val_ds.create_dict_iterator():
        (input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
            audio_pos_ids, attention_mask, gather_index, _, _,
            _, _, _, img_masks, _, _, _, _, _, _,
            _, _, _, _, _, _, _, _, _, _,_) = get_batch_data(batch)
        scores = model.predict(input_ids, position_ids, img_feat, img_pos_feat, audio_feat, audio_pos_ids,
                  attention_mask, gather_index, img_masks)
        bs = scores.shape[0]
        score_vec[n_ex:n_ex + bs] = scores[:,0]
        n_ex += bs
        print(f"{n_ex}/{pair_num ** 2}")
        if n_ex >= pair_num ** 2:
            break

    if not is_parallel or get_rank()==0:
        score_vec = score_vec[:n_ex]
        k = 10
        score_mat = score_vec.reshape((int(math.sqrt(n_ex)), -1))

        max_targets = np.arange(0, int(math.sqrt(n_ex)), dtype=np.int64)
        values, topk_indices = topk(score_mat, 10)
        topk_ind = topk_indices.asnumpy()
        gt_img_j = np.expand_dims(max_targets, 1).repeat(k, axis=1)
        _, rank = np.nonzero(topk_ind == gt_img_j)
        tr_r1 = (rank < 1).sum().item() / int(math.sqrt(n_ex))
        tr_r5 = (rank < 5).sum().item() / int(math.sqrt(n_ex))
        tr_r10 = (rank < 10).sum().item() / int(math.sqrt(n_ex))

        score_mat = score_mat.T
        values, topk_indices = topk(score_mat, 10)
        topk_ind = topk_indices.asnumpy()
        gt_img_j = np.expand_dims(max_targets, 1).repeat(k, axis=1)
        _, rank = np.nonzero(topk_ind == gt_img_j)
        ir_r1 = (rank < 1).sum().item() / int(math.sqrt(n_ex))
        ir_r5 = (rank < 5).sum().item() / int(math.sqrt(n_ex))
        ir_r10 = (rank < 10).sum().item() / int(math.sqrt(n_ex))

        ret_logs = {}
        ret_logs["ir_r1"] = ir_r1
        ret_logs["ir_r5"] = ir_r5
        ret_logs["ir_r10"] = ir_r10
        ret_logs["tr_r1"] = tr_r1
        ret_logs["tr_r5"] = tr_r5
        ret_logs["tr_r10"] = tr_r10
        return ret_logs
    return None

def str2bool(b):
    if b.lower() in ["false"]:
        return False
    return True

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "../..")
    print('project_root:', project_root)
    print('process id:', os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", help='JSON config files')
    parser.add_argument('--output_dir', default="", type=str, help='output directory')
    parser.add_argument('--use_parallel', default=True,
                        type=str2bool, help='use parallel')
    parser.add_argument('--use_txt_out', default=False,
                        type=str2bool, help='use txt out')
    parser.add_argument('--use_video', default=False,
                        type=str2bool, help='use video')
    parser.add_argument("--use_moe", default=False,
                        type=bool, help="use moe")
    
    parser.add_argument("--pretrain_ckpt_file", default=None,
                        type=str, help="pretrain ckpt file path")
    parser.add_argument("--finetune_ckpt_file", default=None,
                        type=str, help="finetune ckpt file path")
    parser.add_argument("--ckpt_size", default=32,
                        type=int, help="distribute ckpt nums")
    parser.add_argument("--ckpt_path", default=None,
                        type=str, help="distribute ckpt path")
    parser.add_argument("--pretrain_ckpt_name", default=None,
                        type=str, help="distribute pretrain ckpt name")
    parser.add_argument("--finetune_ckpt_name", default=None,
                        type=str, help="distribute finetune ckpt name")
    
    args = parse_with_config(parser)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512
    print(args)
    main(args)
