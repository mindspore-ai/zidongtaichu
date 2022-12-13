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
import sys
sys.path.append('./')

import argparse
import os
import time
import mindspore
from mindspore.common.tensor import Tensor
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore import context
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.model import Model

from src.model_mindspore.optim_ms import build_optimizer
from src.model_mindspore.speech_ms import UniterThreeForPretrainingForAdWithLoss
from src.model_mindspore.utils import LearningRate
from src.tools.logger import LOGGER
from src.tools.misc import parse_with_config, set_random_seed
from src.tools.monitor import LossMonitorSingleTask
from src.model_mindspore.cell_wrapper import TrainOneStepWithLossScaleCell
from src.data import data_column, create_audio_dataset

from pathlib2 import Path

def guard_val(val):
    if val is None:
        return Tensor(0).astype(mindspore.int32)
    return val

def main(opts):
    device_id = int(os.getenv('DEVICE_ID'))
    rank_id_str = os.getenv('RANK_ID', '0')
    rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])  # 'RANK_ID': 'job24535502-job-facereidtome-hn-0/1'
    print('rank_id:{}'.format(rank_id), "rank_id str:{}".format(rank_id_str))
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

    print('local_rank:{}, device id:{} start to run...'.format(local_rank, device_id), flush=True)

    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        save_graphs_path=save_graphs_path,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(variable_memory_max_size="30GB")
    context.set_context(reserve_class_name_in_scope=False)

    if opts.use_parallel:
        init()
        LOGGER.info("start init")

        device_num = get_group_size()
        rank = get_rank()
        opts.rank = rank
        LOGGER.info("device_id is {}, rank_id is {}, device_num is {}".format(
            device_id, rank, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            device_num=device_num,
            full_batch=opts.full_batch,
            enable_alltoall=True,
            loss_repeated_mean=True,
            enable_parallel_optimizer=True,
            pipeline_stages=1,
            strategy_ckpt_save_file=strategy_ckpt_save_file)
    else:
        device_num = 1
        rank = 0
        opts.rank = rank

    ds = create_audio_dataset(opts, device_num=device_num)
    dataset_size = ds.get_dataset_size()
    print("=====dataset size: ", dataset_size, flush=True)
    if opts.sink_size > 0:
        new_epoch = opts.epochs * dataset_size // opts.sink_size
        callback_size = opts.sink_size
    else:
        new_epoch = opts.epochs
        callback_size = dataset_size

    net_with_loss = UniterThreeForPretrainingForAdWithLoss(opts.model_config, full_batch=opts.full_batch,
                                                           use_moe=opts.use_moe, opts=opts)

    lr = LearningRate(opts.start_learning_rate, opts.end_learning_rate, opts.warmup_steps, opts.decay_steps)
    optimizer = build_optimizer(net_with_loss, opts, lr)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=opts.init_loss_scale,
                                             scale_factor=opts.loss_scale_factor,
                                             scale_window=opts.scale_window)

    net_with_grads = TrainOneStepWithLossScaleCell(net_with_loss, optimizer, update_cell)

    model = Model(net_with_grads)

    # all cards will save ckpt
    save_steps = opts.save_checkpoint_steps
    ckpt_dir = os.path.join(opts.output_dir, 'ckpt', f"rank_{str(local_rank)}")
    if not os.path.exists(ckpt_dir):
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    config_ck = CheckpointConfig(save_checkpoint_steps=save_steps,
                                 keep_checkpoint_max=10,
                                 integrated_save=False)
    ckpoint_cb = ModelCheckpoint(prefix="OPT_speech",
                                 directory=ckpt_dir,
                                 config=config_ck)
    callback = [TimeMonitor(callback_size), LossMonitorSingleTask(callback_size)]
    callback.append(ckpoint_cb)

    print("=====new_epoch: ", new_epoch, flush=True)

    model.train(new_epoch, ds, callbacks=callback, dataset_sink_mode=True, sink_size=callback_size)


def str2bool(b):
    if b.lower() not in ["false", "true"]:
        raise Exception("Invalid Bool Value")
    if b.lower() in ["false"]:
        return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default="/home/work/user-job-dir/uniter-three/config/" +
                        "pretrain_three_modal_txt_img_audio_config.json",
                        help='JSON config files')
    parser.add_argument("--start_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float,
                        help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=120000, type=int,
                        help="The decay step.")
    parser.add_argument('--use_txt_out', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_video', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--use_parallel', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--data_type', default=2, type=int, help='use txt out')

    parser.add_argument('--audio_dim', default=1024, type=int, help='use txt out')
    parser.add_argument('--img_dim', default=2048, type=int, help='use txt out')
    parser.add_argument('--use_data_fix', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--use_mask_fix', default=True, type=str2bool, help='use txt out')

    parser.add_argument('--name_txt', default="id2len_three.json", type=str, help='use txt out')
    parser.add_argument('--name_img', default="img2len_three.json", type=str, help='use img out')
    parser.add_argument('--name_audio', default="audio2len_three.json", type=str, help='use audio out')

    parser.add_argument("--init_loss_scale", default=65536, type=float, help="")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="")
    parser.add_argument("--scale_window", default=1000, type=float, help="")
    parser.add_argument("--load_ckpt", default=True, type=bool, help="")
    parser.add_argument("--save_checkpoint_steps", default=1000, type=int, help="")
    parser.add_argument("--epochs", default=10, type=int, help="")
    parser.add_argument("--bucket_dir", default="s3://muti-modal/ckpt/", type=str, help="")
    parser.add_argument('--sink_size', default=2, type=int, help='sink size.')
    parser.add_argument("--full_batch", default=False, type=bool, help="")
    parser.add_argument("--use_moe", default=False, type=bool, help="use moe")
    parser.add_argument('--use_vit', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--use_patch', default=True, type=str2bool, help='use txt out')

    args = parse_with_config(parser)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
