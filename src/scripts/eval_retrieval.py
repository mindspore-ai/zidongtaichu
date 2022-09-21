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
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

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
        device_id = int(os.getenv('DEVICE_ID'))
        print('device_id:{}'.format(device_id))
        rank_id_str = os.getenv('RANK_ID', '0')
        rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])
        print('rank_id:{}'.format(rank_id), "rank_id str:{}".format(rank_id_str))
    else:
        device_num = 1
        device_id = int(os.getenv('DEVICE_ID'))
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
        print((f"=====device_num:{device_num} dp:{ParallelConfig.dp} mp:{ParallelConfig.mp} "
               f"enable_optimizer_shard:{ParallelConfig.optimizer_shard} op:{opts.optimizer_shard_size} " 
               f"train_batch_size:{opts.train_batch_size} val_batch_size:{opts.val_batch_size}"))
    else:
        device_num = 1

    # init dataset
    ds = create_dataset(opts, device_num=device_num,
                        token_size=opts.val_batch_size, is_train=False)
    dataset_size = ds.get_dataset_size()
    print("=====dataset size: ", dataset_size, flush=True)

    return rank_id, ds

def load_ckpt(net, ckpt_file):
    if not ckpt_file:
        return
    print('start loading ckpt:', ckpt_file)
    param_dict = load_checkpoint(ckpt_file)
    if param_dict:
        param_not_load = load_param_into_net(net, param_dict)
        print("==========param not load==========:", param_not_load)
        print('load ckpt finished:', ckpt_file)

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

def load_distributed_ckpt(net, rank_id, ckpt_size, ckpt_path, ckpt_name):
    if not ckpt_path or not ckpt_name:
        return
    ckpt_id = rank_id % ckpt_size
    folder = "rank_" + str(ckpt_id)
    ckpt_file = os.path.join(ckpt_path, folder, ckpt_name)
    if not os.path.exists(ckpt_file):
        print(f"ckpt_file:{ckpt_file} does not exists")
        return
    print('start loading distributed ckpt:', ckpt_file)
    param_dict = load_checkpoint(ckpt_file)
    if param_dict:        
        param_not_load = load_param_into_net(net, param_dict)
        print("==========param not load==========:", param_not_load)
        print('load distributed ckpt finished:', ckpt_file)

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

class LearningRate(LearningRateSchedule):
    """ LearningRate """

    def __init__(self,
                 start_learning_rate,
                 end_learning_rate,
                 warmup_steps,
                 decay_steps,
                 power=1.0,
                 use_cosine=True):
        super(LearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(start_learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(start_learning_rate, end_learning_rate, decay_steps, power)
        self.cosine_decay_lr = CosineDecayLR(end_learning_rate, start_learning_rate, decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))
        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()
        self.use_cosine = use_cosine

    def construct(self, global_step):
        """ construct """
        if not self.use_cosine:
            decay_lr = self.decay_lr(global_step)
        else:
            decay_lr = self.cosine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


def guard_val(val):
    """ guard_val """
    if val is None:
        return Tensor(0).astype(ms.int32)
    return val


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
    load_ckpt(net_without_loss, opts.ckpt_file)
    load_pretrain_ckpt(net_without_loss, opts.pretrain_ckpt_file)
    load_distributed_ckpt(net_without_loss, rank_id, opts.ckpt_size, opts.ckpt_path, opts.ckpt_name)
    load_distributed_pretrain_ckpt(net_without_loss, rank_id, opts.ckpt_size, opts.ckpt_path, opts.pretrain_ckpt_name)
    
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
    project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "../../")
    print('project_root:', project_root)
    print('process id:', os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", help='JSON config files')
    parser.add_argument('--output_dir', default="", type=str, help='output directory')
    parser.add_argument('--use_txt_out', default=False,
                        type=str2bool, help='use txt out')
    parser.add_argument('--use_video', default=False,
                        type=str2bool, help='use video')
    parser.add_argument('--use_patch', default=False,
                        type=str2bool, help='use patch')
    parser.add_argument('--path_size', default=32,
                        type=int, help='path size')
    parser.add_argument('--use_parallel', default=True,
                        type=str2bool, help='use parallel')
    parser.add_argument('--data_type', default=2,
                        type=int, help='data type')
    parser.add_argument('--use_data_fix', default=True,
                        type=str2bool, help='use data fix')
    parser.add_argument('--use_mask_fix', default=True,
                        type=str2bool, help='use mask fix')
    parser.add_argument('--name_txt', default="id2len_three.json",
                        type=str, help='txt id2len file')
    parser.add_argument('--name_img', default="img2len_three.json",
                        type=str, help='img img2len file')
    parser.add_argument('--name_audio', default="audio2len_three.json",
                        type=str, help='audio audio2len file')
    parser.add_argument("--init_loss_scale",
                        default=65536, type=float, help="init loss scale")
    parser.add_argument("--loss_scale_factor", default=2,
                        type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=1000,
                        type=float, help="scale window")
    parser.add_argument("--ckpt_file", default=None,
                        type=str, help="ckpt file path to load")
    parser.add_argument("--pretrain_ckpt_file", default=None,
                        type=str, help="pretrain ckpt file path to load")
    parser.add_argument("--ckpt_size", default=32,
                        type=int, help="distribute ckpt nums to load")
    parser.add_argument("--ckpt_path", default=None,
                        type=str, help="distribute ckpt path to load")
    parser.add_argument("--ckpt_name", default=None,
                        type=str, help="distribute ckpt name to load")
    parser.add_argument("--pretrain_ckpt_name", default=None,
                        type=str, help="distribute pretrain ckpt name to load")
    parser.add_argument("--save_checkpoint_steps",
                        default=0, type=int, help="save checkpoint steps")
    parser.add_argument("--epochs", default=3,
                        type=int, help="epochs")
    parser.add_argument("--full_batch", default=True,
                        type=bool, help="use full batch")
    parser.add_argument("--use_moe", default=False,
                        type=bool, help="use moe")

    args = parse_with_config(parser)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
