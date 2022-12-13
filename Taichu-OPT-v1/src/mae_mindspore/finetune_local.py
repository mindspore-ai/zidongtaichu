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
import numpy as np

from mindspore import nn
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.communication.management import init, get_group_size, get_rank

from src.vit import Vit
from src.loss import get_loss
from src.logger import get_logger
from src.datasets import get_dataset
from src.monitor import StateMonitor
from src.lr_generator import LearningRate
from src.eval_engine import get_eval_engine
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
    train_dataset = get_dataset(args)
    per_step_size = train_dataset.get_dataset_size()
    if args.per_step_size:
        per_step_size = args.per_step_size
    args.logger.info("Create training dataset finish, data size:{}".format(per_step_size))

    # evaluation dataset
    eval_dataset = get_dataset(args, is_train=False)
    if args.device_num == 1:
        args.eval_engine = ""

    net = Vit(batch_size=args.batch_size, patch_size=args.patch_size,
              image_size=args.train_image_size, dropout=args.dropout,
              num_classes=args.num_classes, **args.model_config)

    eval_engine = get_eval_engine(args.eval_engine, net, eval_dataset, args)

    # loss scale
    manager_loss_scale = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)

    # define lr_schedule
    lr_schedule = LearningRate(
        args.start_learning_rate, args.end_learning_rate,
        args.epoch, args.warmup_epochs, per_step_size
    )

    # define optimizer
    optimizer = nn.AdamWeightDecay(net.trainable_params(), learning_rate=lr_schedule, weight_decay=0.05)

    # define loss
    if not args.use_label_smooth:
        args.label_smooth_factor = 0.0
    vit_loss = get_loss(loss_name=args.loss_name, args=args)

    # define callback
    state_cb = StateMonitor(data_size=per_step_size,
                            tot_batch_size=args.batch_size * device_num,
                            eval_interval=args.eval_interval,
                            eval_offset=args.eval_offset,
                            eval_engine=eval_engine,
                            logger=args.logger.info)
    callback = [state_cb, ]

    # model config
    save_ckpt_feq = args.save_ckpt_epochs * per_step_size
    config_ck = CheckpointConfig(save_checkpoint_steps=save_ckpt_feq,
                                 keep_checkpoint_max=1,
                                 integrated_save=False)
    ckpoint_cb = ModelCheckpoint(prefix=args.prefix,
                                 directory=args.ckpt_save_dir,
                                 config=config_ck)
    callback += [ckpoint_cb]

    # load finetune ckpt
    if os.path.exists(args.finetune_ckpt):
        params_dict = load_checkpoint(args.finetune_ckpt)
        net_not_load = net.init_weights(params_dict)
        args.logger.info(f"===============net_not_load================{net_not_load}")

    # define Model and begin training
    model = Model(net, loss_fn=vit_loss, optimizer=optimizer,
                  metrics=eval_engine.metric, eval_network=eval_engine.eval_network,
                  loss_scale_manager=manager_loss_scale, amp_level="O3")
    eval_engine.set_model(model)
    t0 = time.time()
    # equal to model._init(dataset, sink_size=per_step_size)
    eval_engine.compile(sink_size=per_step_size)
    t1 = time.time()
    args.logger.info('compile time used={:.2f}s'.format(t1 - t0))
    model.train(args.epoch, train_dataset, callbacks=callback,
                dataset_sink_mode=args.sink_mode, sink_size=per_step_size)
    last_metric = 'last_metric[{}]'.format(state_cb.best_acc)
    args.logger.info(last_metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config', default="./config/finetune-vit-base-p16.yaml",
        help='YAML config files')
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use_parallel.")
    parser.add_argument("--per_step_size", default=2, type=int, help="per_step_size.")
    args_ = parse_with_config(parser)
    if args_.eval_offset < 0:
        args_.eval_offset = args_.epoch % args_.eval_interval

    main(args_)
