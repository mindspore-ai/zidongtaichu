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
import argparse
import os
import time

from mindspore import context
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from pathlib2 import Path

import sys
sys.path.append('.')
from src.data import create_dataset
from src.model_mindspore.cell_wrapper import TrainOneStepWithLossScaleCell
from src.model_mindspore.optim_ms import build_optimizer
from src.model_mindspore.vqa import UniterThreeForPretrainingForVQAFinetune
from src.model_mindspore.utils import LearningRate
from src.model_mindspore.utils import LossSummaryCallbackLocal
from src.model_mindspore.parallel_transformer import ParallelConfig
from src.config.config import *
from src.tools.misc import parse_with_config, set_random_seed
from src.tools.monitor import LossMonitorSingleTask

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

    strategy_ckpt_save_file = save_graphs_path + "strategy" + str(local_rank) + ".ckpt"

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
        ParallelConfig.dp = device_num
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
    ds = create_dataset(opts, device_num=device_num,
                        token_size=opts.train_batch_size, is_train=True)
    dataset_size = ds.get_dataset_size()
    print("dataset size: ", dataset_size, flush=True)
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

def load_ckpt(net, ckpt_file):
    if not ckpt_file:
        return
    print(f"start loading ckpt:{ckpt_file}")
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
        param_not_load = load_param_into_net(net, new_params_dict)
        print("param not load:", param_not_load)
    print(f"end loading ckpt:{ckpt_file}")

def load_vit_ckpt(net, vit_ckpt_file):
    if not vit_ckpt_file:
        return
    print(f"start loading img-encoder:{vit_ckpt_file}")
    vit_dict = load_checkpoint(vit_ckpt_file)
    if vit_dict:
        param_dict = {}
        for k,v in vit_dict.items():
            if k.startswith('encoder.'):        #vit 448
                param_dict['uniter.img_embeddings.vit.' + k[8:]] = v
        param_not_load = load_param_into_net(net.uniter.img_embeddings.vit, param_dict)
        print("param not load:", param_not_load)
    print(f"end loading img-encoder:{vit_ckpt_file}")

def main(opts):
    (local_rank, _, callback_size, _, _, _,
     new_epoch, ds, dataset_size) = init_env(opts)
    net_with_loss = UniterThreeForPretrainingForVQAFinetune(opts.model_config, img_dim=IMG_DIM,
                                                            img_label_dim=IMG_LABEL_DIM,
                                                            audio_dim=AUDIO_DIM, audio_label_dim=AUDIO_LABEL_DIM,
                                                            use_txt_out=opts.use_txt_out, use_video=opts.use_video,
                                                            full_batch=opts.full_batch, use_moe=opts.use_moe,
                                                            is_parallel=opts.use_parallel)
    print(net_with_loss)
    
    load_ckpt(net_with_loss, opts.ckpt_file.strip())
    load_vit_ckpt(net_with_loss, opts.vit_ckpt_file.strip())
    
    if not opts.decay_steps:
        opts.decay_steps = opts.decay_epochs * dataset_size
    lr = LearningRate(opts.start_learning_rate, opts.end_learning_rate, opts.warmup_steps, opts.decay_steps)
    optimizer = build_optimizer(net_with_loss, opts, lr)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=opts.init_loss_scale,
                                             scale_factor=opts.loss_scale_factor,
                                             scale_window=opts.scale_window)
    net_with_grads = TrainOneStepWithLossScaleCell(net_with_loss, optimizer=optimizer,
                                                   scale_sense=update_cell)
    callback = [TimeMonitor(callback_size), LossMonitorSingleTask(callback_size,verbose=False)]
    if local_rank == 0:
        if not opts.save_checkpoint_steps:
            opts.save_checkpoint_steps = dataset_size
        config_ck = CheckpointConfig(save_checkpoint_steps=opts.save_checkpoint_steps,
                                    keep_checkpoint_max=10,
                                    integrated_save=False)
        ckpt_dir = os.path.join(opts.output_dir, 'ckpt', f"rank_{str(local_rank)}")
        if not os.path.exists(ckpt_dir):
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        ckpoint_cb = ModelCheckpoint(prefix="OPT_vqa",
                                    directory=ckpt_dir,
                                    config=config_ck)
        callback.append(ckpoint_cb)
        summary_dir = os.path.join(opts.output_dir, 'loss_summary')
        callback.append(LossSummaryCallbackLocal(summary_dir=summary_dir,
                                                 local_rank=local_rank,
                                                 has_trained_epoch=0,
                                                 has_trained_step=0))
    model = Model(net_with_grads)
    print("start_training...")
    model.train(new_epoch, ds, callbacks=callback, dataset_sink_mode=opts.dataset_sink_mode, sink_size=callback_size)

def str2bool(b):
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception("Invalid Bool Value")
    return output

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
    print('project root:', project_root)
    print('process id:', os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", help='JSON config files')
    parser.add_argument('--vit_ckpt_file', default="", type=str)
    parser.add_argument('--ckpt_file', default="", type=str)
    
    parser.add_argument("--start_learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float,
                        help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=0, type=int,
                        help="lr decay steps.")
    parser.add_argument("--decay_epochs", default=10, type=int, help="lr decay epochs.")
    parser.add_argument("--train_batch_size", default=10, type=int,
                        help="The decay step.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="The decay step.")
    
    parser.add_argument('--callback_size', default=100, type=int, help='callback size.')
    parser.add_argument('--dataset_sink_mode', default=False, type=str2bool, help='dataset sink mode')
    parser.add_argument('--use_txt_out', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_video', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_patch', default=False, type=str2bool, help='use patch')
    parser.add_argument('--patch_size', default=32, type=int, help='path size')
    parser.add_argument('--use_parallel', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--data_type', default=2, type=int, help='use txt out')
    parser.add_argument('--output_dir', default="", type=str, help='output directory')
    parser.add_argument("--init_loss_scale", default=65536, type=float, help="")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="")
    parser.add_argument("--scale_window", default=1000, type=float, help="")
    parser.add_argument("--load_ckpt", default=True, type=bool, help="")
    parser.add_argument("--save_checkpoint_steps", default=0, type=int, help="")
    parser.add_argument('--data_url', required=True, default=None, help='URL of data.')
    parser.add_argument('--train_url', required=True, default=None, help='URL of data.')
    parser.add_argument("--bucket_dir", default="s3://muti-modal/ckpt/", type=str, help="")
    parser.add_argument("--full_batch", default=False, type=bool, help="")
    parser.add_argument("--use_moe", default=False, type=bool, help="use moe")
    parser.add_argument("--mode", default="train", type=str)

    args = parse_with_config(parser)
    print(args)
    main(args)