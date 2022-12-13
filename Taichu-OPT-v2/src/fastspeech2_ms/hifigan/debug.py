import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json

from env import AttrDict, build_env
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, \
    GeneratorWithLossCell, DiscriminatorWithLossCell

import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import load_checkpoint, load_param_into_net
from mindspore import context, ops

import numpy as np
import random

def train(rank, a, h):

    os.makedirs(a.checkpoint_path, exist_ok=True)
    print("checkpoints directory : ", a.checkpoint_path)
    
    generator = Generator(h)
    # mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    
    steps = 0
    cp_g = 'gen_mindspore.ckpt'
    cp_sd = 'msd_mindspore.ckpt'
    cp_pd = 'mpd_mindspore.ckpt'
    if cp_g is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g)
        state_dict_msd = load_checkpoint(cp_sd)
        state_dict_mpd = load_checkpoint(cp_pd)
        # load_param_into_net(generator, state_dict_g, strict_load=True)
        # load_param_into_net(mpd, state_dict_mpd, strict_load=False)
        # load_param_into_net(msd, state_dict_msd, strict_load=False)
        # mindspore.save_checkpoint(msd, 'msd_mindspore.ckpt')
        steps = 0
        last_epoch = -1
    netD_with_criterion = DiscriminatorWithLossCell(generator, mpd, msd)
    netG_with_criterion = GeneratorWithLossCell(generator, mpd, msd)
    
    optim_g = nn.AdamWeightDecay(params=generator.trainable_params(), learning_rate=h.learning_rate, 
                               beta1=h.adam_b1, beta2=h.adam_b2)
    optim_d = nn.AdamWeightDecay(params=msd.trainable_params(), 
                               learning_rate=h.learning_rate, beta1=h.adam_b1, beta2=h.adam_b2)
    TrainOneStepCellForD = nn.TrainOneStepCell(netD_with_criterion, optim_d)
    TrainOneStepCellForG = nn.TrainOneStepCell(netG_with_criterion, optim_g)

    TrainOneStepCellForD.set_train()
    TrainOneStepCellForG.set_train()
    
    for epoch in range(1):
        start = time.time()
        print("Epoch: {}".format(epoch+1))

        for i in range(10):
            x = mindspore.Tensor(np.random.rand(16,80,32), dtype=mindspore.dtype.float32)
            y = mindspore.Tensor(np.random.rand(16,8192), dtype=mindspore.dtype.float32)
            print(y)
            start_b = time.time()
            # x, y, _ = batch
            y = ops.expand_dims(y, 1)
            
            loss_D = TrainOneStepCellForD(x, y)
            loss_G = TrainOneStepCellForG(x, y)

            if steps % a.stdout_interval == 0:
                print('Steps : {:d}, Gen Loss Total : {:4.3f}, Dis Loss Total : {:4.3f}, s/b : {:4.3f}'.
                        format(steps, loss_G.asnumpy(), loss_D.asnumpy(), time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}.ckpt".format(a.checkpoint_path, steps)
                    mindspore.save_checkpoint(generator, checkpoint_path)
                    checkpoint_path = "{}/do_{:08d}.ckpt".format(a.checkpoint_path, steps)
                    mindspore.save_checkpoint({'mpd':mpd, 'msd':msd}, checkpoint_path)

            steps += 1
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='/root/small_data/speech_data/AISHELL3/train/wav_22050')
    parser.add_argument('--input_mels_dir', default='/root/small_data/speech_data/AISHELL3/train/mel')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='config_v1.json')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=1, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=True, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=2)
    np.random.seed(0)
    random.seed(0)
    mindspore.set_seed(0)
    
    train(0, a, h)


if __name__ == '__main__':
    main()
