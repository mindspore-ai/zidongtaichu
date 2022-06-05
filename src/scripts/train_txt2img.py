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
import sys
import time
import json

import mindspore
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore.context import ParallelMode
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from pathlib2 import Path

sys.path.append('./')
from src.data import data_column, create_dataset
from src.model_mindspore.cell_wrapper import TrainOneStepWithLossScaleCell
from src.model_mindspore.optim_ms import build_optimizer
from src.model_mindspore.t2i import UniterThreeForPretrainingForT2Ifinetune
from src.model_mindspore.utils import LearningRate
from src.model_mindspore.utils import LossSummaryCallbackLocal
from src.tools.const import IMG_LABEL_DIM, AUDIO_LABEL_DIM
from src.tools.logger import LOGGER
from src.tools.misc import parse_with_config, set_random_seed
from src.tools.monitor import LossMonitorSingleTask, ValidMonitor
from src.data.pretrain_three_data import create_three_dataloaders

project_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


def guard_val(val):
    if val is None:
        return Tensor(0).astype(mindspore.int32)
    return val


# Masked Language Modeling (MLM);
# Masked Region Feature Regression (MRFR);
# Masked Region Classification (MRC);
# Masked Region Classification with KL-divergence (MRC-kl);
# Image-Text Matching (ITM).
def main(opts):
    device_id = int(os.getenv('DEVICE_ID'))
    rank_id_str = os.getenv('RANK_ID', '0')
    rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])  # 'RANK_ID': 'job24535502-job-facereidtome-hn-0/1'
    print('rank_id:{}'.format(rank_id), "rank_id str:{}".format(rank_id_str))
    local_rank = rank_id
    print('local_rank:{}, device id:{}'.format(local_rank, device_id))
    profiling_path = f'/cache/{local_rank}-graphs/'
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
    mindspore.common.set_seed(7)
    context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL)
    context.set_context(mode=context.GRAPH_MODE,
                        save_graphs=False,
                        device_target="Ascend",
                        device_id=device_id)
    context.set_context(variable_memory_max_size="30GB")
    context.set_context(reserve_class_name_in_scope=False)
    init()

    device_num = get_group_size()
    rank = get_rank()
    opts.rank = rank
    print("device_id is {}, rank_id is {}, device_num is {}".format(
        device_id, rank, device_num))

    ds = create_dataset(opts, device_num=device_num,
                        token_size=opts.train_batch_size,
                        is_train=True, print_time=False)

    net_with_loss = UniterThreeForPretrainingForT2Ifinetune(opts.model_config,
                                                            img_dim=opts.img_dim, audio_dim=opts.audio_dim,
                                                            use_txt_out=opts.use_txt_out, use_video=opts.use_video,
                                                            full_batch=opts.full_batch, use_moe=opts.use_moe,
                                                            args=opts)

    ckpt_file = opts.ckpt_file
    print(ckpt_file)
    if ckpt_file != "":
        params_dict = load_checkpoint(ckpt_file)
        net_not_load = load_param_into_net(net_with_loss, params_dict)
        print("===============net_not_load================", net_not_load)

    net_with_loss = _VirtualDatasetCell(net_with_loss)

    lr = LearningRate(opts.start_learning_rate, opts.end_learning_rate, opts.warmup_steps, opts.decay_steps)
    optimizer = build_optimizer(net_with_loss, opts, lr)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=opts.init_loss_scale,
                                             scale_factor=opts.loss_scale_factor,
                                             scale_window=opts.scale_window)
    net_with_grads = TrainOneStepWithLossScaleCell(net_with_loss, optimizer=optimizer,
                                                   scale_sense=update_cell)
    # all cards will save ckpt
    save_steps = opts.save_checkpoint_steps
    ckpt_dir = os.path.join(opts.output_dir, 'ckpt', f"rank_{str(local_rank)}")
    if not os.path.exists(ckpt_dir):
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # save opts
    with open(os.path.join(opts.output_dir, 'opts.log'), 'w') as f:
        f.write(str(opts))
        f.close()
    # sink train
    sink_size = 1000
    callback_size = 50
    dataset_size = ds.get_dataset_size()
    new_epoch = opts.epochs * dataset_size // (opts.train_batch_size * sink_size)
    print(f"=====dataset size: {dataset_size}, sink size: {sink_size}, epochs: {opts.epochs}, new epochs: {new_epoch}=====", flush=True)
    print(f"================Total Iteration:{sink_size * new_epoch}===============================", flush=True)
    callback = [TimeMonitor(callback_size), LossMonitorSingleTask(callback_size, False)]
    # save validation info
    if local_rank == 0 and opts.ids_val_path is not "":
        print(f"Loading Valid callback... [valid_step:{opts.valid_steps}]")
        log_file = os.path.join(opts.output_dir, f'validation.log')
        validloaders, _ = create_three_dataloaders(opts.ids_val_path, opts.val_datasets, False, opts, device_num)
        validloader = validloaders["ftT2I"]
        validmonitor = ValidMonitor(opts.valid_steps, net_with_loss, validloader, log_file)
        callback.append(validmonitor)

    if local_rank == 0:
        # Setting CKPT callback
        print(f"Loading CKPT callback... [save_steps-{save_steps}, ckpt_dir-{ckpt_dir}]")
        config_ck = CheckpointConfig(save_checkpoint_steps=save_steps,
                                     keep_checkpoint_max=2,
                                     integrated_save=False)
        ckpoint_cb = ModelCheckpoint(prefix="OPT_txt2img",
                                     directory=ckpt_dir,
                                     config=config_ck)
        callback.append(ckpoint_cb)
        # Setting Summary callback
        print(f"Loading Summary callback... [summary_dir-{os.path.join(opts.output_dir, 'loss_summary')}]")
        summary_dir = os.path.join(opts.output_dir, 'loss_summary')
        callback.append(LossSummaryCallbackLocal(summary_dir=summary_dir,
                                                 local_rank=0,
                                                 has_trained_epoch=0,
                                                 has_trained_step=0))
    model = Model(net_with_grads)
    model.train(new_epoch, ds, callbacks=callback, dataset_sink_mode=True, sink_size=sink_size)


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
    parser.add_argument('--use_parallel', default=False, type=str2bool, help='use txt out')
    parser.add_argument('--data_type', default=2, type=int, help='use txt out')

    parser.add_argument('--audio_dim', default=512, type=int, help='use txt out')
    parser.add_argument('--img_dim', default=2048, type=int, help='use txt out')
    parser.add_argument('--use_data_fix', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--use_mask_fix', default=True, type=str2bool, help='use txt out')

    parser.add_argument('--name_txt', default="id2len_three.json", type=str, help='use txt out')
    parser.add_argument('--name_img', default="img2len_three.json", type=str, help='use img out')
    parser.add_argument('--name_audio', default="audio2len_three.json", type=str, help='use audio out')
    parser.add_argument('--output_dir', default="", type=str, help='use audio out')

    parser.add_argument("--init_loss_scale", default=65536, type=float, help="")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="")
    parser.add_argument("--scale_window", default=1000, type=float, help="")
    parser.add_argument("--load_ckpt", default=True, type=bool, help="")
    parser.add_argument("--save_checkpoint_steps", default=1000, type=int, help="")
    parser.add_argument("--epochs", default=0, type=int, help="")
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--ckpt_file', default="", type=str, help='use txt out')
    parser.add_argument("--bucket_dir", default="s3://muti-modal/ckpt/", type=str, help="")
    parser.add_argument('--sink_size', default=2, type=int, help='sink size.')
    parser.add_argument("--full_batch", default=False, type=bool, help="")
    parser.add_argument("--use_moe", default=False, type=bool, help="use moe")
    parser.add_argument('--use_vit', default=False, type=str2bool, help='use txt out')

    args = parse_with_config(parser)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
