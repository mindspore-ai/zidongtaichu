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
sys.path.append('/')

import argparse
import os
import time
import json
import tqdm
import numpy as np
import mindspore
from mindspore.common.tensor import Tensor
from mindspore.communication.management import init, get_group_size, get_rank
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.tools.logger import LOGGER
from src.tools.misc import parse_with_config, set_random_seed
from data import create_three_dataloaders
from src.model_mindspore.speech_ms import UniterThreeForPretrainingForAdEval
from data import get_batch_data_audio_eval
from src.fastspeech2_mindspore import hifigan

from pathlib2 import Path
from scipy.io import wavfile

def guard_val(val):
    if val is None:
        return Tensor(0).astype(mindspore.int32)
    return val


def get_vocoder(speaker):
    '''
    mindspore;done;useful
    '''
    name = "HiFi-GAN"
    with open("src/fastspeech2_mindspore/hifigan/config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    if speaker == "LJSpeech":
        ckpt = load_checkpoint("src/fastspeech2_mindspore/hifigan/generator_LJSpeech.ckpt")
    elif speaker == "universal":
        ckpt = load_checkpoint("src/fastspeech2_mindspore/hifigan/generator_universal.ckpt")

    parm_not_load = load_param_into_net(vocoder, ckpt, strict_load=True)
    print(parm_not_load)
    return vocoder


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

    test_loaders, datalen = create_three_dataloaders(opts.ids_val_path, opts.val_datasets, False,
                                                  opts, device_num=device_num)
    test_loader = test_loaders['adTextEval']

    ckpt_file = opts.ckpt_file
    print(ckpt_file)
    if ckpt_file == "":
        modified_params_dict = None
    else:
        params_dict = load_checkpoint(ckpt_file)

        modified_params_dict = {}
        for k, v in params_dict.items():
            if 'txt_output.tfm_decoder' in k:
                modified_k = k.replace('txt_output.tfm_decoder', 'txt_output.tfm_decoder.decoder.tfm_decoder')
                v.name = v.name.replace('txt_output.tfm_decoder', 'txt_output.tfm_decoder.decoder.tfm_decoder')
                modified_v = v
                modified_params_dict[modified_k] = modified_v
            else:
                modified_params_dict[k] = v

    net_without_loss = UniterThreeForPretrainingForAdEval(opts.model_config, full_batch=opts.full_batch,
                                                           use_moe=opts.use_moe, opts=opts)

    if modified_params_dict:
        net_not_load = load_param_into_net(net_without_loss, modified_params_dict)
        print("===============net_not_load================", net_not_load)

    validate_ad(net_without_loss, test_loader, opts)


def validate_ad(model, test_loader, opts):
    """
     validate_td
    """
    LOGGER.info("start running Audio Decoder validation...")
    MAX_WAV_VALUE = 32768.0

    total = len(test_loader.dataset)
    num = 0

    pbar = tqdm.tqdm(total=total)

    vocoder = get_vocoder("universal")

    for batch in test_loader:

        ids = batch['ids']

        input_ids, position_ids, attention_mask, \
        mel_targets, duration_targets, speakers, \
        texts, src_lens, mel_lens, audio_max_text_len, \
        audio_max_mel_len, pitch_targets, energy_targets = get_batch_data_audio_eval(batch)

        postnet_mel_predictions, mel_lens = model(input_ids, position_ids, attention_mask,
                    mel_targets, duration_targets, speakers,
                    texts, src_lens, mel_lens, audio_max_text_len,
                    audio_max_mel_len, pitch_targets, energy_targets)

        mel_save_dir = os.path.join(opts.output_result_dir, "mel_save")
        wave_save_dir = os.path.join(opts.output_result_dir, "wav_save")

        if not os.path.exists(mel_save_dir):
            os.makedirs(mel_save_dir, exist_ok=True)

        if not os.path.exists(wave_save_dir):
            os.makedirs(wave_save_dir, exist_ok=True)

        for i in range(len(ids)):
            id_ = ids[i]
            postnet_mel_prediction = postnet_mel_predictions[i].asnumpy()
            mel_len = mel_lens[i].asnumpy()

            # mel save
            mel = postnet_mel_prediction[:mel_len.item(), :]
            path_mel = os.path.join(mel_save_dir, id_.replace("/", "_"))
            np.save(path_mel, mel)

            time_start = time.time()
            # wave save
            mel = mindspore.Tensor(mel.T, dtype=mindspore.dtype.float32)
            mel = mindspore.ops.expand_dims(mel, 0)
            print(mel.shape)
            y_g_hat = vocoder(mel)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.asnumpy().astype('int16')
            print("time: {:.3f}s".format(time.time() - time_start))

            path_wave = os.path.join(wave_save_dir, id_.replace("/", "_") + ".wav")

            wavfile.write(path_wave, 22050, audio)

            pbar.update(1)

            num += 1
            if num >= total:
                break

        if num >= total:
            break

    print("Finish")


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
    parser.add_argument('--ckpt_file', default="", type=str, help='use txt out')
    parser.add_argument('--output_result_dir', default="", type=str, help='use txt out')
    parser.add_argument('--use_vit', default=True, type=str2bool, help='use txt out')
    parser.add_argument('--use_patch', default=True, type=str2bool, help='use txt out')

    args = parse_with_config(parser)

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
