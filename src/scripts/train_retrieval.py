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
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint, SummaryCollector
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR, CosineDecayLR
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")))
from src.config.config import *
from src.data import create_dataset, get_batch_data
from src.data.utils import pad_sequence
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
            opts.train_batch_size = int(opts.train_batch_size * ParallelConfig.dp)        
        print((f"=====device_num:{device_num} dp:{ParallelConfig.dp} mp:{ParallelConfig.mp} "
               f"enable_optimizer_shard:{ParallelConfig.optimizer_shard} op:{opts.optimizer_shard_size} " 
               f"train_batch_size:{opts.train_batch_size} val_batch_size:{opts.val_batch_size}"))
    else:
        device_num = 1

    # init dataset
    ds = create_dataset(opts, device_num=device_num,
                        token_size=opts.train_batch_size, is_train=True)
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

    return local_rank, rank_id, callback_size, new_epoch, ds, dataset_size

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
    ckpt_id = rank_id % ckpt_size
    folder = "rank_" + str(ckpt_id)
    if not ckpt_path or not ckpt_name:
        return
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
    ckpt_id = rank_id % ckpt_size
    folder = "rank_" + str(ckpt_id)
    if not ckpt_path or not ckpt_name:
        return
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

class Profiler(ms.Callback):
    def __init__(self, start_step, stop_step):
        super(Profiler, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        self.profiler = ms.Profiler(start_profile=False)
    def step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            self.profiler.start()
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step:
            self.profiler.stop()
    def end(self, run_context):
        self.profiler.analyse()

def main(opts):
    # init
    (local_rank, rank_id, callback_size, epochs, ds, dataset_size) = init_env(opts)

    # create model
    net_with_loss = UniterThreeForPretrainingForRetFinetune(opts.model_config, img_dim=IMG_DIM,
                                                            img_label_dim=IMG_LABEL_DIM,
                                                            audio_dim=AUDIO_DIM, audio_label_dim=AUDIO_LABEL_DIM,
                                                            use_txt_out=opts.use_txt_out, use_video=opts.use_video,
                                                            full_batch=opts.full_batch, use_moe=opts.use_moe,
                                                            args=opts, is_parallel=opts.use_parallel)

    # learning rate and optimizer
    lr = LearningRate(opts.start_learning_rate, opts.end_learning_rate, opts.warmup_steps, opts.decay_steps)
    optimizer = build_optimizer(net_with_loss, opts, lr)
    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=opts.init_loss_scale,
                                             scale_factor=opts.loss_scale_factor,
                                             scale_window=opts.scale_window)

    # build net with grads
    net_with_grads = ParallelTrainOneStepWithLossScaleCell(net_with_loss, optimizer=optimizer,
                                                           scale_sense=update_cell, parallel_config=ParallelConfig)

    # set callbacks
    callbacks = [TimeMonitor(callback_size), LossMonitor(callback_size)]
    
    # path to save ckpt
    if not opts.save_checkpoint_steps:
        opts.save_checkpoint_steps = dataset_size
    ckpt_dir = os.path.join(opts.output_dir, "ckpt", f"rank_{str(local_rank)}")
    if not os.path.exists(ckpt_dir):
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    
    # set checkpoint callback
    config_ck = CheckpointConfig(save_checkpoint_steps=opts.save_checkpoint_steps,
                                    keep_checkpoint_max=1,
                                    integrated_save=False)
    ckpoint_cb = ModelCheckpoint(prefix="OPT_ret",
                                 directory=ckpt_dir,
                                 config=config_ck)
    callbacks.append(ckpoint_cb)
    
    # modify summary callback
    if opts.save_summary:
        specified = {"collect_metric": True, "collect_graph": True, "collect_dataset_graph": True}
        summary_collector = SummaryCollector(summary_dir=os.path.join(opts.output_dir, 'summary'), 
        collect_specified_data=specified, collect_freq=1, keep_default_action=False, collect_tensor_freq=200)
        callbacks.append(summary_collector)
    
    # callbacks.append(Profiler(1, 5))
    
    # build model
    model = Model(net_with_grads)
    
    # load ckpt
    load_pretrain_ckpt(net_with_loss, opts.pretrain_ckpt_file)
    load_finetune_ckpt(net_with_loss, opts.finetune_ckpt_file)
    if opts.dataset_sink_mode:
        print("start building...")
        model.build(train_dataset=ds, sink_size=callback_size, epoch=epochs)
    load_distributed_pretrain_ckpt(net_with_loss, rank_id, opts.ckpt_size, opts.ckpt_path, opts.pretrain_ckpt_name)
    load_distributed_finetune_ckpt(net_with_loss, rank_id, opts.ckpt_size, opts.ckpt_path, opts.finetune_ckpt_name)

    print("start training...")
    model.train(epochs, ds, callbacks=callbacks, dataset_sink_mode=opts.dataset_sink_mode, sink_size=callback_size)

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
    parser.add_argument('--callback_size', default=1, type=int, help='callback size.')
    parser.add_argument('--dataset_sink_mode', default=False, type=str2bool, help='dataset sink mode')
    parser.add_argument('--save_summary', default=False, type=str2bool, help='save summary')
    parser.add_argument("--start_learning_rate", default=2.5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-8, type=float,
                        help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=20000, type=int,
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
