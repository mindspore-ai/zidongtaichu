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

import os, time
import yaml
import argparse
import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.train.model import Model
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor

from src.utils.get_loss import get_loss
from src.utils.get_model import get_model
from src.utils.get_logger import get_logger_dist as get_logger
from src.utils.get_dataset import get_dataset_dist as get_dataset
from src.utils.get_modeloss import get_modeloss
from src.utils.monitor import TimeMonitorV2
from src.utils.lr_manager import LearningRateV2


def get_config():
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('-p', '--opt_path', required=True, help="Path of config file")
    # The following two arguments are necessary for Cloud AND negligible for Local.
    parser.add_argument('--data_url', default=None, help='Location of data.')
    parser.add_argument('--train_url', default=None, help='Location of training outputs.')
    args = parser.parse_args()
    assert os.path.exists(args.opt_path), f"Fail to load {args.opt_path}"

    path = args.opt_path
    print("Loading opt file: ", path)
    assert os.path.exists(path), f"{path} must exists!"
    with open(path, 'r') as f:
        opt = yaml.load(f)
    return opt, args

class CustomWithEvalCell(nn.Cell):
    def __init__(self, model, loss_fn):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)

        self._model = model
        self._loss_fn = loss_fn

    def construct(self, x):
        output = self._model(x, is_training=False)
        x_rec = output['x_rec']
        vq_output = output['vq_output']

        return self._loss_fn(x, x_rec), vq_output['loss'], x, x_rec

# import ipdb
if __name__ == '__main__':
    opt, args = get_config()
    mindspore.common.set_seed(7)
    context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)
    init()

    # initialize model and loss function
    net = get_model(opt['model'])
    loss = get_loss(opt['loss'])

    if opt['train']['ckpt'] is not None:
        param_dict = load_checkpoint(opt['train']['ckpt'])
        start_epoch = int(param_dict['epoch'].asnumpy())
        start_iteration = int(param_dict['tot_iter'].asnumpy())
        load_param_into_net(net, param_dict)
        exp_path = os.path.dirname(opt['train']['ckpt'])
        exp_name = os.path.basename(exp_path)
        logger = get_logger(os.path.join(exp_path, f'{time.time()}.log'),
                            exp_name, ifstdout=opt['train']['std_out'])
        logger.info(f"Successfully load ckpt: {opt['train']['ckpt']}.")
    else:
        start_epoch = 0
        start_iteration = 0
        exp_name = opt['train']['exp_name']
        exp_path = os.path.join(os.path.abspath(opt['train']['exp_path']), exp_name)
        if get_rank() == 0:
            os.makedirs(exp_path, exist_ok=True)
            os.makedirs(os.path.join(exp_path, 'rec_img'), exist_ok=True)
            os.makedirs(os.path.join(exp_path, 'gt_img'), exist_ok=True)
        logger = get_logger(os.path.join(exp_path, f'{time.time()}.log'), exp_name, ifstdout=opt['train']['std_out'])
        if opt['train']['pretrain'] is not None:
            param_dict = load_checkpoint(opt['train']['pretrain'])
            load_param_into_net(net, param_dict)
            logger.info(f"Successfully load ckpt: {opt['train']['pretrain']}.")
        else:
            logger.info(f"Do not load any checkpoint.")

    logger.info(f"exp_path: {exp_path}")
    logger.info(f"exp_name: {exp_name}")
    logger.info(f"Model Parameters: {str(opt['model'])}")
    logger.info(f"Dataset Parameters: {str(opt['dataset'])}")
    logger.info(f"Loss Parameters: {str(opt['loss'])}")
    logger.info(f"Training Parameters: {str(opt['train'])}")
    logger.info(f"Current rank: {get_rank()}, group size: {get_group_size()}")
    logger.info(f"start_epoch: {start_epoch}, start_iter: {start_iteration}.")

    # load dataset
    trainset, validset = get_dataset(opt['dataset'])
    step_per_epoch = trainset.get_dataset_size() if opt['train']['sink_size'] == -1 else opt['train']['sink_size']
    tot_epoch = (opt['train']['num_epochs'] * trainset.get_dataset_size()) // step_per_epoch
    logger.info(f"Setting epoch: {opt['train']['num_epochs']}, Total epoch: {tot_epoch}")
    logger.info(f"Total iteraions: {tot_epoch * step_per_epoch}")
    logger.info(f"Step per epoch: {step_per_epoch} with sink size: {opt['train']['sink_size']}.")

    # get model, lr_scheduler and optimizer
    modeloss_fn = get_modeloss(opt)
    modelwithloss = modeloss_fn(net, loss)

    logger.info(f"Adopting warmup_cosine_annealing learning rate scheduler...")
    lr_scheduler = LearningRateV2(start_learning_rate=opt['train']['lr'] * (1 - start_epoch / tot_epoch),
                                 end_learning_rate=0.00001, warmup_steps=10000,
                                 decay_steps=step_per_epoch * (tot_epoch - start_epoch))

    if 'optimize' not in opt['train'].keys() or opt['train']['optimize'].lower() == 'adam':
        logger.info(f"Adopting ADAM optimizer...")
        optimizer = nn.Adam(net.trainable_params(), learning_rate=lr_scheduler, beta1=0.9, beta2=0.99) # beta2=0.95 is better for large model
    elif opt['train']['optimize'].lower() == 'sgd':
        logger.info(f"Adopting SGD optimizer...")
        optimizer = nn.SGD(net.trainable_params(), learning_rate=lr_scheduler)
    elif opt['train']['optimize'].lower() == 'adamweightdecay':
        logger.info(f"Adopting AdamWeightDecay optimizer...")
        optimizer = nn.AdamWeightDecay(params=net.trainable_params(), learning_rate=lr_scheduler)
    else:
        logger.info(f"Fail to adopt optimizer: {opt['train']['optimize']}!")
        logger.info(f"Adopting ADAM optimizer...")
        optimizer = nn.Adam(net.trainable_params(), learning_rate=lr_scheduler, beta1=0.9, beta2=0.99)

    # model to train
    manager_loss_scale = nn.DynamicLossScaleUpdateCell(loss_scale_value=opt['train']['loss_scale'], scale_factor=2, scale_window=1000)
    train_model = nn.TrainOneStepWithLossScaleCell(modelwithloss, optimizer, manager_loss_scale)
    # model to eval
    valid_model = CustomWithEvalCell(net, loss)
    valid_model.set_train(False)

    Monitor_time = TimeMonitorV2(1, valid_model, validset, exp_path, logger)
    Monitor_loss = LossMonitor(per_print_times=10)

    config_ckpt = CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=30)
    assert os.path.exists(exp_path), f"{exp_path} does not exists!!!"
    ckpt_cb = ModelCheckpoint(prefix=opt['model']['name'], directory=exp_path, config=config_ckpt)
    logger.info(f"CKPT is save in: {exp_path}")
    callbacks = [Monitor_time, Monitor_loss, ckpt_cb]

    model = Model(train_model, keep_batchnorm_fp32=opt['train']['keep_batchnorm_fp32'])
    logger.info(f"======================Start training!============================")
    model.train(tot_epoch, trainset, callbacks=callbacks,
                dataset_sink_mode=True, sink_size=opt['train']['sink_size'])
