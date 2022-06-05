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
import json
import yaml
import argparse
import mindspore
from PIL import Image
import numpy as np
import mindspore.ops as ops
from mindspore import context
from mindspore import Tensor, Parameter
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.communication import init, get_rank, get_group_size

from tqdm import tqdm
from src.vqvae_mindspore.src.utils.get_model import get_model

def get_config():
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('-p', '--opt_path', help="Path of config file")
    parser.add_argument('--ckpt', type=str, required=True)
    # method1
    parser.add_argument('--cspath_json_file', type=str, default=None)
    parser.add_argument('--cpath_dir', type=str, default=None)
    parser.add_argument('--spath_dir', type=str, default=None)
    # method2
    parser.add_argument('--tokens_path', type=str, default=None)
    # arguments for cloud, negligible for LOCAL
    parser.add_argument('--data_url', default=None, help='Location of data.')
    parser.add_argument('--train_url', default=None, help='Location of training outputs.')
    args = parser.parse_args()
    assert os.path.exists(args.opt_path), f"OPT file \"{args.opt_path}\" does not exist!!!!"

    path = args.opt_path
    print("Loading opt file: ", path)
    assert os.path.exists(path), f"{path} must exists!"
    with open(path, 'r') as f:
        opt = yaml.load(f)
    return opt, args


def get_visualize_img(img): # img: [C H W], np.array
    show_x = np.clip(img, a_min=0, a_max=1)
    show_x = np.transpose(show_x, (1, 2, 0)) * 255.
    show_x = Image.fromarray(show_x.astype(np.uint8)).convert('RGB')
    return show_x

if __name__ == '__main__':
    opt, args = get_config()
    mindspore.common.set_seed(7)
    context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)
    init()

    net = get_model(opt['model'])
    param_dict = load_checkpoint(args.ckpt)
    load_param_into_net(net, param_dict)
    print(f"Successfully load ckpt: {args.ckpt}.", flush=True)

    if args.cspath_json_file is not None and args.tokens_path is None:
        cspaths = json.load(open(args.cspath_json_file, 'r'))
        print(f"Loaded {len(cspaths)} cspaths.", flush=True)
        for (cpath, spath) in tqdm(cspaths):
            if args.cpath_dir is not None:
                cpath = os.path.join(args.cpath_dir, cpath)
            if args.spath_dir is not None:
                spath = os.path.join(args.spath_dir, spath)

            code = np.load(cpath).reshape((1, 32, 32))
            xrec = net.get_xrec_from_codes(mindspore.Tensor(code))
            xrec = ops.clip_by_value(xrec, clip_value_min=Tensor(0),
                                     clip_value_max=Tensor(1))
            xrec = xrec.asnumpy()[0]
            show_x = np.transpose(xrec, (1, 2, 0)) * 255.
            show_x = Image.fromarray(show_x.astype(np.uint8)).convert('RGB')
            show_x.save(spath)

    elif args.cspath_json_file is None and args.tokens_path is not None:
        save_path = args.tokens_path + '_visualize'
        os.makedirs(save_path, exist_ok=True)
        print(f"Visualize tokens in {args.tokens_path} in {save_path}")
        for cfile in tqdm(os.listdir(args.tokens_path)):
            code = np.load(os.path.join(args.tokens_path, cfile)).reshape((1, 16, 16))
            xrec = net.get_xrec_from_codes(mindspore.Tensor(code))
            xrec = ops.clip_by_value(xrec, clip_value_min=Tensor(0),
                                     clip_value_max=Tensor(1))
            xrec = xrec.asnumpy()[0]
            show_x = np.transpose(xrec, (1, 2, 0)) * 255.
            show_x = Image.fromarray(show_x.astype(np.uint8)).convert('RGB')
            show_x.save(os.path.join(save_path, cfile.replace('.npy', '.png')))