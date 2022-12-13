# Copyright 2021 Huawei Technologies Co., Ltd
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
import sys
sys.path.append("./")
import argparse
from src.config import config as C
import mindspore
import mindspore as ms
from mindspore.common.tensor import Tensor
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore import context
from mindspore.nn import TrainOneStepCell
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.data import create_dataset
from src.model_mindspore.cell_wrapper import ParallelTrainOneStepWithLossScaleCell, ParallelAccumTrainOneStepWithLossScaleCell
from src.model_mindspore.optim_ms import build_optimizer
from src.model_mindspore.parallel_transformer import ParallelConfig
from src.model_mindspore.pretrain_two_ms import UniterTwoForPretrainingWithLoss
from src.model_mindspore.utils import LearningRate
from src.tools.logger import LOGGER, add_log_to_file
from src.tools.misc import parse_with_config, set_random_seed
from src.tools.monitor import LossMonitor
from pathlib2 import Path


project_root = os.path.abspath(os.path.dirname(
    os.path.realpath(__file__)) + os.path.sep + "..")
print('project_root:', project_root)


def guard_val(val):
    if val is None:
        return Tensor(0).astype(mindspore.int32)
    return val

def init_config(opts):

    C.USE_LARGE_DATA = getattr(opts, 'use_large_data', False)

    C.IMG_DIM = getattr(opts, 'img_dim', 768)
    C.IMG_SIZE = opts.image_size
    C.IMG_PATCH_SIZE = opts.patch_size

    C.MAX_IMG_LEN = (C.IMG_SIZE // C.IMG_PATCH_SIZE)**2 + 1
    C.MAX_IMG_TEXT_LEN = C.MAX_FULL_TEXT_LEN + C.MAX_IMG_LEN
    C.MAX_FULL_LEN = C.MAX_FULL_TEXT_LEN + C.MAX_IMG_LEN + C.MAX_AUDIO_LEN

    print(f"USE_LARGE_DATA:{C.USE_LARGE_DATA}")
    print(f"IMG_DIM:{C.IMG_DIM} IMG_SIZE:{C.IMG_SIZE} IMG_PATCH_SIZE:{C.IMG_PATCH_SIZE}")
    print(f"MAX_IMG_LEN:{C.MAX_IMG_LEN} MAX_IMG_TEXT_LEN:{C.MAX_IMG_TEXT_LEN}  MAX_FULL_LEN:{C.MAX_FULL_LEN}")

def init_env(opts):

    """ init_env """
    if opts.use_parallel:
        device_id = int(os.getenv('DEVICE_ID'))
        LOGGER.info(f'device_id:{device_id}')
        rank_id_str = os.getenv('RANK_ID', '0')
        # 'RANK_ID': 'job24535502-job-facereidtome-hn-0/1'
        rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])
        LOGGER.info(f'rank_id:{rank_id} rank_id str:{rank_id_str}')
    else:
        device_num = 1
        device_id = int(os.getenv('DEVICE_ID'))
        rank = 0
        rank_id = 0
        opts.rank = rank

    LOGGER.info(f'output_dir: {opts.output_dir}')
    log_dir = os.path.join(opts.output_dir, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    add_log_to_file(os.path.join(log_dir, f"log_{rank_id}.txt"))

    local_rank = rank_id
    LOGGER.info(f'local_rank:{local_rank}, device id:{device_id}')
    profiling_path = os.path.join(opts.output_dir, f'cache/{local_rank}-graphs/')
    if not os.path.exists(profiling_path):
        Path(profiling_path).mkdir(parents=True, exist_ok=True)
    save_graphs_path = os.path.join(profiling_path, "graph")
    if not os.path.exists(save_graphs_path):
        Path(save_graphs_path).mkdir(parents=True, exist_ok=True)

    strategy_ckpt_save_file = save_graphs_path + \
                              "strategy" + str(local_rank) + ".ckpt"

    os.environ['HCCL_CONNECT_TIMEOUT'] = "6000"

    os.system('ulimit -s 102400')
    set_random_seed(opts.seed)

    LOGGER.info(f'local_rank:{local_rank}, device id:{device_id} start to run...')

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
        LOGGER.info(f"device_id is {device_id}, rank_id is {rank}, device_num is {device_num}")
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num)
    else:
        device_num = 1

    ParallelConfig.mp = 1
    ParallelConfig.optimizer_shard = False

    ds = create_dataset(opts, device_num=device_num)
    dataset_size = ds.get_dataset_size()
    LOGGER.info(f"=====dataset size: {dataset_size}")
    if opts.sink_size > 0:
        new_epoch = opts.epochs * dataset_size // opts.sink_size
        callback_size = opts.sink_size
    else:
        new_epoch = opts.epochs
        # callback_size = dataset_size
        callback_size = 1

    ParallelConfig.dp = device_num // ParallelConfig.mp

    if opts.full_batch:
        opts.train_batch_size = opts.train_batch_size * ParallelConfig.dp
        opts.val_batch_size = opts.train_batch_size * ParallelConfig.dp

    LOGGER.info(f"=====device_num:{device_num} dp:{ParallelConfig.dp} mp:{ParallelConfig.mp} "
                f"train_batch_size:{opts.train_batch_size} val_batch_size:{opts.val_batch_size}")

    return local_rank, rank_id, callback_size, strategy_ckpt_save_file, device_id, device_num, new_epoch, ds

def numel(shape):
    total = 1
    for val in shape:
        total *= val
    return total

def load_ckpt(net_with_grads, ckpt_file):
    if not ckpt_file or len(ckpt_file) == 0:
        print("not load ckpt")
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
        # new_params_dict["uniter.img_embeddings.img_linear.weight"] = new_params_dict["feat_regress.weight"]
        # new_params_dict["uniter.audio_embeddings.audio_linear.weight"] = new_params_dict["audio_feat_regress.weight"]
        new_params_dict["uniter.embeddings.word_embeddings.embedding_table"] = new_params_dict["cls.predictions.decoder.weight"]
        net_not_load = load_param_into_net(net_with_grads, new_params_dict)
        print("===============net_not_load================", net_not_load)
    # print("init model......................................")
    # net_with_grads.init_output()
    print('load ckpt:', ckpt_file)

def set_param(net_param_dict, ms_full_name, numpy_data):
    old_param = net_param_dict[ms_full_name]
    new_param = ms.Tensor(numpy_data, old_param.data.dtype)
    old_param.set_data(new_param)

def load_proj_ckpt(net_with_grads):
    import pickle
    import numpy as np

    ckpt_path = "/mnt/sfs_turbo/cn_clip_ckpt/clip_cn_vit-b-16-proj.ckpt"
    with open(ckpt_path, 'rb') as ckpt_fp:
        param_dict = pickle.load(ckpt_fp)

    net_param = net_with_grads.parameters_dict()

    set_param(net_param, "text_proj.weight", param_dict["module.text_projection"].T)
    set_param(net_param, "vision_proj.weight", param_dict["module.visual.proj"].T)
    set_param(net_param, "text_proj.bias", np.zeros((512,)))
    set_param(net_param, "vision_proj.bias", np.zeros((512,)))
    set_param(net_param, "clip_loss.clip_temp", param_dict["module.logit_scale"])

    print(f"load {ckpt_path} success")

class StopAtStep(ms.Callback):
    def __init__(self, start_step, stop_step):
        super(StopAtStep, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        self.profiler = ms.Profiler(start_profile=False, output_path='/mnt/sfs_turbo/xxzhu/profile')

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
            self.profiler.analyse()
            exit()

    def end(self, run_context):
        self.profiler.analyse()

def main(opts):

    # init
    init_config(opts)

    (local_rank, rank_id, callback_size, strategy_ckpt_save_file, device_id, device_num,
     new_epoch, ds) = init_env(opts)

    fintune_itm = getattr(opts, "finetune_itm", False)
    print(f"fintune_itm: {fintune_itm}")

    opt_model = UniterTwoForPretrainingWithLoss

    net_with_loss = opt_model(opts.model_config, opts)

    if getattr(opts, "load_proj_ckpt", False):
        print("load_proj_ckpt True")
        load_proj_ckpt(net_with_loss)
    else:
        print("load_proj_ckpt False")

    # load_ckpt(net_with_loss, opts.ckpt_file)

    if rank_id == 0:
        LOGGER.info("model have {} paramerters in total".format(sum(numel(x.shape) for x in net_with_loss.get_parameters())))
        LOGGER.info("model text have {} paramerters in total".format(sum(numel(x.shape) for x in net_with_loss.uniter.embeddings.get_parameters())))
        LOGGER.info("model img have {} paramerters in total".format(sum(numel(x.shape) for x in net_with_loss.uniter.img_embeddings.get_parameters())))
        LOGGER.info("model cross have {} paramerters in total".format(sum(numel(x.shape) for x in net_with_loss.uniter.encoder.get_parameters())))

    net_with_loss = _VirtualDatasetCell(net_with_loss)

    lr = LearningRate(opts.start_learning_rate,
                      opts.end_learning_rate, opts.warmup_steps, opts.decay_steps)
    optimizer = build_optimizer(net_with_loss, opts, lr)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=opts.init_loss_scale,
                                             scale_factor=opts.loss_scale_factor,
                                             scale_window=opts.scale_window)

    accum_grad = getattr(opts, "accum_grad", 1)
    if accum_grad > 1:
        net_with_grads = ParallelAccumTrainOneStepWithLossScaleCell(net_with_loss, optimizer,
                                                                    update_cell, accumulation_steps=accum_grad)
    elif opts.use_parallel:
        net_with_grads = ParallelTrainOneStepWithLossScaleCell(net_with_loss, optimizer=optimizer,
                                                           scale_sense=update_cell, enable_global_norm=True,
                                                           clip_norm=opts.grad_norm, parallel_config=ParallelConfig)
    else:
        net_with_grads = TrainOneStepCell(net_with_loss, optimizer)
    # all cards will save ckpty
    save_steps = opts.save_checkpoint_steps
    ckpt_dir = os.path.join(opts.output_dir, "train/ckpt", f"rank_{str(local_rank)}")
    if not os.path.exists(ckpt_dir):
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)


    if rank_id == 0:

        config_ck = CheckpointConfig(save_checkpoint_steps=save_steps,
                                     keep_checkpoint_max=3,
                                     integrated_save=False)
        ckpoint_cb = ModelCheckpoint(prefix="OPT",
                                     directory=ckpt_dir,
                                     config=config_ck)
        callback = [TimeMonitor(callback_size), LossMonitor(callback_size, opts.is_two)]
        callback.append(ckpoint_cb)
    else:
        callback = []

    model = Model(net_with_grads)

    print('=====start training...=====')
    model.train(new_epoch, ds, callbacks=callback, dataset_sink_mode=False, sink_size=callback_size)


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
    # parser.add_argument('--audio_dim', default=1024,
    #                     type=int, help='audio dim')
    # parser.add_argument('--img_dim', default=2048,
    #                     type=int, help='img dim')
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
                        default=2000, type=int, help="save checkpoint steps")
    parser.add_argument("--epochs", default=100,
                        type=int, help="epochs")
    parser.add_argument('--sink_size', default=0,
                        type=int, help='sink size.')
    parser.add_argument("--full_batch", default=False,
                        type=bool, help="use full batch")
    parser.add_argument("--use_moe", default=False,
                        type=bool, help="use moe")
    parser.add_argument("--is_two", default=False,
                        type=bool, help="two model")
    parser.add_argument("--use_pipeline", default=False,
                        type=bool, help="use pipeline")
    args = parse_with_config(parser)

    main(args)
