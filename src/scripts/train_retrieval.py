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
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from src.config.config import *
from src.data import create_dataset, get_batch_data
from src.model_mindspore.parallel_transformer import ParallelConfig
from src.model_mindspore.cell_wrapper import ParallelTrainOneStepWithLossScaleCell
from src.model_mindspore.retrieval_ms import UniterThreeForPretrainingForRetFinetune, \
                                    UniterThreeForPretrainingForRetFinetuneEval
from src.model_mindspore.optim_ms import build_optimizer
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
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.DATA_PARALLEL,
            gradients_mean=False,
            device_num=device_num,
            full_batch=opts.full_batch,
            enable_alltoall=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=False,
            pipeline_stages=1,
            strategy_ckpt_save_file=strategy_ckpt_save_file)
    else:
        device_num = 1
    ParallelConfig.dp = device_num
    ds = create_dataset(opts, device_num=device_num,
                        token_size=opts.train_batch_size, is_train=not opts.eval_only)
    dataset_size = ds.get_dataset_size()
    print("=====dataset size: ", dataset_size, flush=True)
    if opts.dataset_sink_mode:
        if opts.callback_size > 0:
            new_epoch = opts.epochs * dataset_size // opts.callback_size
            callback_size = opts.callback_size
        else:
            new_epoch = opts.epochs
            callback_size = dataset_size
    else:
        new_epoch = opts.epochs
        callback_size = opts.callback_size

    return local_rank, rank_id, callback_size, strategy_ckpt_save_file, device_id, device_num, new_epoch, ds, dataset_size

def load_ckpt(net_with_grads, ckpt_file):
    if not ckpt_file:
        return
    print('load ckpt:', ckpt_file)
    params_dict = load_checkpoint(ckpt_file)
    if params_dict:
        new_params_dict = {}
        for key in params_dict.keys():
            if key.find("txt_output.tfm_decoder") >= 0:
                key_new = key[:22] + ".decoder.tfm_decoder" + key[22:]
                new_params_dict[key_new] = params_dict[key]
            new_params_dict[key] = params_dict[key]
        new_params_dict["uniter.img_embeddings.img_linear.weight"] = new_params_dict["feat_regress.weight"]
        new_params_dict["uniter.audio_embeddings.audio_linear.weight"] = new_params_dict["audio_feat_regress.weight"]
        new_params_dict["uniter.embeddings.word_embeddings.embedding_table"] = new_params_dict["cls.predictions.decoder.weight"]
        param_not_load = load_param_into_net(net_with_grads, new_params_dict)
        print("param_not_load:", param_not_load)
    print("init model......................................")
    net_with_grads.init_output()
    print('load ckpt:', ckpt_file)

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
    (local_rank, rank_id, callback_size, strategy_ckpt_save_file, device_id, device_num,
     new_epoch, ds, dataset_size) = init_env(opts)
    # eval
    if opts.eval_only:
        net_without_loss = UniterThreeForPretrainingForRetFinetuneEval(opts.model_config, img_dim=IMG_DIM,
                                                                img_label_dim=IMG_LABEL_DIM,
                                                                audio_dim=AUDIO_DIM, audio_label_dim=AUDIO_LABEL_DIM,
                                                                use_txt_out=opts.use_txt_out, use_video=opts.use_video,
                                                                full_batch=opts.full_batch, use_moe=opts.use_moe,
                                                                       is_parallel=opts.use_parallel)
        ckpt_file = opts.ckpt_file
        load_ckpt(net_without_loss, ckpt_file)
        model = Model(net_without_loss)
        ids = json.load(open(opts.ids_val_path,'r'))
        print("retrieval dataset's length is: ", len(ids))
        log = validate_itm_matching(model, ds, len(ids), is_parallel=opts.use_parallel)
        print(log)
        return
    # create model
    net_with_loss = UniterThreeForPretrainingForRetFinetune(opts.model_config, img_dim=IMG_DIM,
                                                            img_label_dim=IMG_LABEL_DIM,
                                                            audio_dim=AUDIO_DIM, audio_label_dim=AUDIO_LABEL_DIM,
                                                            use_txt_out=opts.use_txt_out, use_video=opts.use_video,
                                                            full_batch=opts.full_batch, use_moe=opts.use_moe,
                                                                       is_parallel=opts.use_parallel)
    # load ckpt
    ckpt_file = opts.ckpt_file
    load_ckpt(net_with_loss, ckpt_file)

    # learning rate and optimizer
    lr = LearningRate(opts.start_learning_rate, opts.end_learning_rate, opts.warmup_steps, opts.decay_steps)
    optimizer = build_optimizer(net_with_loss, opts, lr)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=opts.init_loss_scale,
                                             scale_factor=opts.loss_scale_factor,
                                             scale_window=opts.scale_window)

    # build net with grads
    net_with_grads = ParallelTrainOneStepWithLossScaleCell(net_with_loss, optimizer=optimizer,
                                                           scale_sense=update_cell, parallel_config=ParallelConfig)

    # path to save ckpt
    if not opts.save_checkpoint_steps:
        opts.save_checkpoint_steps = dataset_size
    ckpt_dir = os.path.join(opts.output_dir, "ckpt", f"rank_{str(local_rank)}")
    if not os.path.exists(ckpt_dir):
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    sleep_time = int(rank_id) * 1.5
    print("=====sleep time is, ", sleep_time)

    # set callbacks
    if rank_id == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=opts.save_checkpoint_steps,
                                     keep_checkpoint_max=1,
                                     integrated_save=False)
        ckpoint_cb = ModelCheckpoint(prefix="OPT_ret",
                                     directory=ckpt_dir,
                                     config=config_ck)
        callback = [TimeMonitor(callback_size), LossMonitor(callback_size)]
        callback.append(ckpoint_cb)
    else:
        callback = []
    # train
    model = Model(net_with_grads)
    print("start_training...")
    model.train(new_epoch, ds, callbacks=callback, dataset_sink_mode=opts.dataset_sink_mode, sink_size=callback_size)

def validate_itm_matching(model, val_ds, pair_num=1000, is_parallel = True):
    topk = ops.TopK()
    print("start running ITM validation...")
    score_vec = Tensor(np.zeros((pair_num**2,)), ms.float32)
    n_ex = 0
    for batch in val_ds.create_dict_iterator():
        (input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
            audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
            txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
            mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
            ma_neg_index, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
            txt_masks, img_token_gts, img_token_masks, images, images_mask,
            taskId) = get_batch_data(batch)

        scores = model.predict(input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
            audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
            txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
            mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
            ma_neg_index, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
            txt_masks, img_token_gts, img_token_masks, images, images_mask,
            taskId)
        bs = scores.shape[0]
        score_vec[n_ex:n_ex + bs] = scores[:,0]
        n_ex += bs
        if n_ex >= pair_num ** 2:
            break

    if not is_parallel or get_rank()==0:
        print("===========n_ex=", n_ex)
        score_vec = score_vec[:n_ex]
        print(score_vec.shape)
        print(score_vec)
        k = 10
        score_mat = score_vec.reshape((int(math.sqrt(n_ex)), -1))

        print(score_mat)
        print("........................",score_mat.dtype,score_mat.shape,int(math.sqrt(n_ex)))
        max_targets = np.arange(0, int(math.sqrt(n_ex)), dtype=np.int64)
        values, topk_indices = topk(score_mat, 10)
        topk_ind = topk_indices.asnumpy()
        gt_img_j = np.expand_dims(max_targets, 1).repeat(k, axis=1)
        _, rank = np.nonzero(topk_ind == gt_img_j)
        tr_r1 = (rank < 1).sum().item() / int(math.sqrt(n_ex))
        tr_r5 = (rank < 5).sum().item() / int(math.sqrt(n_ex))
        tr_r10 = (rank < 10).sum().item() / int(math.sqrt(n_ex))
        print(tr_r1, tr_r5, tr_r10)

        score_mat = score_mat.T
        values, topk_indices = topk(score_mat, 10)
        topk_ind = topk_indices.asnumpy()
        gt_img_j = np.expand_dims(max_targets, 1).repeat(k, axis=1)
        _, rank = np.nonzero(topk_ind == gt_img_j)
        ir_r1 = (rank < 1).sum().item() / int(math.sqrt(n_ex))
        ir_r5 = (rank < 5).sum().item() / int(math.sqrt(n_ex))
        ir_r10 = (rank < 10).sum().item() / int(math.sqrt(n_ex))
        print(ir_r1, ir_r5, ir_r10)

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
    project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
    print('project_root:', project_root)
    print('process id:', os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", help='JSON config files')
    parser.add_argument('--output_dir', default="", type=str, help='output directory')
    parser.add_argument('--callback_size', default=100, type=int, help='callback size.')
    parser.add_argument('--dataset_sink_mode', default=False, type=str2bool, help='dataset sink mode')
    parser.add_argument("--eval_only", default=False, type=str2bool,
                        help="eval only?")
    parser.add_argument("--start_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float,
                        help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=120000, type=int,
                        help="The decay step.")
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
    parser.add_argument("--save_checkpoint_steps",
                        default=0, type=int, help="save checkpoint steps")
    parser.add_argument("--epochs", default=10,
                        type=int, help="epochs")
    parser.add_argument("--full_batch", default=False,
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
