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
import mindspore
from mindspore.common.tensor import Tensor
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore import context
from mindspore.nn import TrainOneStepCell
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from pathlib2 import Path

from config.config import *
from data import create_dataset
from src.model_mindspore.cell_wrapper import ParallelTrainOneStepWithLossScaleCell
from src.model_mindspore.optim_ms import build_optimizer
from src.model_mindspore.parallel_transformer import ParallelConfig
from src.model_mindspore.pretrain_ms import UniterThreeForPretrainingWithLoss
from src.model_mindspore.utils import LearningRate
from src.tools.logger import LOGGER
from src.tools.misc import parse_with_config, set_random_seed
from src.tools.monitor import LossMonitor

project_root = os.path.abspath(os.path.dirname(
    os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


def guard_val(val):
    if val is None:
        return Tensor(0).astype(mindspore.int32)
    return val


def init_env(opts):
    """ init_env """
    if opts.use_parallel:
        device_id = int(os.getenv('DEVICE_ID'))
        print('device_id:{}'.format(device_id))
        rank_id_str = os.getenv('RANK_ID', '0')
        # 'RANK_ID': 'job24535502-job-facereidtome-hn-0/1'
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
        LOGGER.info("start init")

        device_num = get_group_size()
        rank = get_rank()
        opts.rank = rank
        print("device_id is {}, rank_id is {}, device_num is {}".format(
            device_id, rank, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num)
    else:
        device_num = 1

    ParallelConfig.dp = device_num
    ds = create_dataset(opts, device_num=device_num,
                        token_size=opts.train_batch_size)
    dataset_size = ds.get_dataset_size()
    print("=====dataset size: ", dataset_size, flush=True)
    if opts.sink_size > 0:
        new_epoch = opts.epochs * dataset_size // opts.sink_size
        callback_size = opts.sink_size
    else:
        new_epoch = opts.epochs
        # callback_size = dataset_size
        callback_size = 100

    return local_rank, rank_id, callback_size, strategy_ckpt_save_file, device_id, device_num, new_epoch, ds


def main(opts):
    # init
    (local_rank, rank_id, callback_size, strategy_ckpt_save_file, device_id, device_num,
     new_epoch, ds) = init_env(opts)

    net_with_loss = UniterThreeForPretrainingWithLoss(opts.model_config, img_dim=IMG_DIM,
                                                      img_label_dim=IMG_LABEL_DIM,
                                                      audio_dim=AUDIO_DIM, audio_label_dim=AUDIO_LABEL_DIM,
                                                      use_txt_out=opts.use_txt_out, use_video=opts.use_video,
                                                      full_batch=opts.full_batch, use_moe=opts.use_moe,
                                                      is_parallel= opts.use_parallel)
    net_with_loss = _VirtualDatasetCell(net_with_loss)

    lr = LearningRate(opts.start_learning_rate,
                      opts.end_learning_rate, opts.warmup_steps, opts.decay_steps)
    optimizer = build_optimizer(net_with_loss, opts, lr)

    # load ckpt
    params_dict = None
    ckpt_file = opts.ckpt_file
    if ckpt_file is not None:
        LOGGER.info("rank_%d: start loading %s.", rank_id, ckpt_file)
        params_dict = load_checkpoint(ckpt_file)
        LOGGER.info("rank_%d: end loading %s.", rank_id, ckpt_file)
    vit_ckpt_file = opts.vit_ckpt_file
    if vit_ckpt_file is not None:
        LOGGER.info("rank_%d: start loading img-encoder %s.", rank_id, vit_ckpt_file)
        vit_dict = load_checkpoint(vit_ckpt_file)
        params_dict = {}
        for k,v in vit_dict.items():
            if k.startswith('encoder.'):        #vit 448
                params_dict['uniter.img_embeddings.vit.' + k[8:]] = v
            # print(k)
            # params_dict['uniter.img_embeddings.vit.' + k] = v
        LOGGER.info("rank_%d: end loading img-encoder %s.", rank_id, vit_ckpt_file)

    if params_dict:
        net_not_load = load_param_into_net(net_with_loss, params_dict)
        opt_not_load = load_param_into_net(optimizer, params_dict)
        if rank_id == 0:
            print("===============net_not_load================", net_not_load)
        # print("===============opt_not_load================", opt_not_load)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=opts.init_loss_scale,
                                             scale_factor=opts.loss_scale_factor,
                                             scale_window=opts.scale_window)
    if opts.use_parallel:
        net_with_grads = ParallelTrainOneStepWithLossScaleCell(net_with_loss, optimizer=optimizer,
                                                           scale_sense=update_cell, parallel_config=ParallelConfig)
    else:
        net_with_grads = TrainOneStepCell(net_with_loss, optimizer)
    # all cards will save ckpt
    save_steps = opts.save_checkpoint_steps
    ckpt_dir = os.path.join(opts.output_dir, "train/ckpt", f"rank_{str(local_rank)}")
    if not os.path.exists(ckpt_dir):
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    sleep_time = int(rank_id) * 1.5
    print("=====sleep time is, ", sleep_time)


    if rank_id == 0:

        config_ck = CheckpointConfig(save_checkpoint_steps=save_steps,
                                     keep_checkpoint_max=3,
                                     integrated_save=False)
        ckpoint_cb = ModelCheckpoint(prefix="OPT",
                                     directory=ckpt_dir,
                                     config=config_ck)

        callback = [TimeMonitor(callback_size), LossMonitor(callback_size)]
        callback.append(ckpoint_cb)
    else:
        callback = []

    model = Model(net_with_grads)


    print('=====start training...=====')
    model.train(new_epoch, ds, callbacks=callback,
                dataset_sink_mode=False, sink_size=callback_size)


def str2bool(b):
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception("Invalid Bool Value")
    return output

if __name__ == "__main__":

    default_path = "/home/work/user-job-dir/uniter-three/config/pretrain_three_modal_txt_img_audio_config.json"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=default_path,
                        help='JSON config files')
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
    parser.add_argument("--vit_ckpt_file", default=None,
                        type=str, help="vit ckpt file path to load")
    parser.add_argument("--save_checkpoint_steps",
                        default=5000, type=int, help="save checkpoint steps")
    parser.add_argument("--epochs", default=100,
                        type=int, help="epochs")
    parser.add_argument('--sink_size', default=0,
                        type=int, help='sink size.')
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
