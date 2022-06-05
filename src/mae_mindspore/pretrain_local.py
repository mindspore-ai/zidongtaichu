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
import argparse
import numpy as np

from mindspore import nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.communication.management import init, get_group_size, get_rank

from src.logger import get_logger
from src.monitor import LossMonitor
from src.datasets import get_dataset
from src.mae_vit import PreTrainMAEVit
from src.lr_generator import LearningRate
from src.helper import parse_with_config, str2bool


def context_init(args):
    np.random.seed(args.seed)
    set_seed(args.seed)
    rank_id = 0
    device_num = 1
    if args.use_parallel:
        init()
        device_id = int(os.getenv('DEVICE_ID'))  # 0 ~ 7
        rank_id = get_rank()  # local_rank
        device_num = get_group_size()  # world_size
        print("device_id is {}, rank_id is {}, device_num is {}".format(device_id, rank_id, device_num), flush=True)
        args.context["device_id"] = device_id
        context.set_context(**args.context)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.DATA_PARALLEL,
            device_num=device_num,
            gradients_mean=True)
    else:
        context.set_context(**args.context)

    context.set_context(variable_memory_max_size="30GB")
    context.set_context(reserve_class_name_in_scope=False)
    return rank_id, device_num


def main(args):
    local_rank, device_num = context_init(args)
    args.device_num = device_num
    args.local_rank = local_rank
    args.logger = get_logger(args.ckpt_save_dir, rank=args.local_rank)
    args.logger.info("model config: {}".format(args))

    # train dataset
    dataset = get_dataset(args)
    per_step_size = dataset.get_dataset_size()
    if args.per_step_size:
        per_step_size = args.per_step_size
    args.logger.info("Create training dataset finish, data size:{}".format(per_step_size))

    net = PreTrainMAEVit(batch_size=args.batch_size, patch_size=args.patch_size, image_size=args.train_image_size,
                         encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
                         encoder_num_heads=args.encoder_num_heads, decoder_num_heads=args.decoder_num_heads,
                         encoder_dim=args.encoder_dim, decoder_dim=args.decoder_dim,
                         mlp_ratio=args.mlp_ratio, masking_ratio=args.masking_ratio)
    # loss scale
    manager_loss_scale = nn.DynamicLossScaleUpdateCell(
        loss_scale_value=args.loss_scale, scale_factor=args.scale_factor,
        scale_window=args.scale_window)

    # define lr_schedule
    lr_schedule = LearningRate(
        args.start_learning_rate, args.end_learning_rate,
        args.epoch, args.warmup_epochs, per_step_size
    )

    # define optimizer
    optimizer = nn.AdamWeightDecay(net.trainable_params(),
                                   learning_rate=lr_schedule,
                                   beta1=0.9, beta2=0.95,
                                   weight_decay=0.05)

    # define model
    train_model = nn.TrainOneStepWithLossScaleCell(net, optimizer, manager_loss_scale)

    # define callback
    callback = [LossMonitor(args.cb_size, logger=args.logger.info, device_num=args.device_num), ]

    # model config
    save_ckpt_feq = args.save_ckpt_epochs * per_step_size
    if local_rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=save_ckpt_feq,
                                     keep_checkpoint_max=1,
                                     integrated_save=False)
        ckpoint_cb = ModelCheckpoint(prefix=args.prefix,
                                     directory=args.ckpt_save_dir,
                                     config=config_ck)
        callback += [ckpoint_cb]

    # define Model and begin training
    model = Model(train_model)
    model.train(args.epoch, dataset, callbacks=callback,
                dataset_sink_mode=args.sink_mode, sink_size=per_step_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config', default="./config/maevit-base-p16.yaml", help='YAML config files')
    parser.add_argument('--use_parallel', default=True, type=str2bool, help='use_parallel.')
    parser.add_argument("--per_step_size", default=2, type=int, help="per_step_size.")

    args_ = parse_with_config(parser)

    main(args_)
