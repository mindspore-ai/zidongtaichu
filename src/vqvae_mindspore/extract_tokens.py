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
import yaml
import argparse
from PIL import Image
import numpy as np
import mindspore
from mindspore import context
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.communication import init, get_rank, get_group_size

from tqdm import tqdm
from src.utils.get_loss import get_loss
from src.utils.get_model import get_model
from src.utils.get_dataset import get_TokenDataset

def get_config():
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('-p', '--opt_path', help="Path of config file")
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--cspath_json_file', type=str, required=True)
    parser.add_argument('--cpath_dir', type=str, default=None)
    parser.add_argument('--spath_dir', type=str, default=None)
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
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)

    # get model and load ckpt
    net = get_model(opt['model'])
    param_dict = load_checkpoint(args.ckpt)
    load_param_into_net(net, param_dict)
    print(f"Successfully load ckpt: {args.ckpt}.")

    loss = get_loss(opt['loss'])
    dataset = get_TokenDataset(dist=False,
                               data_opt=opt['dataset'],
                               cspath_json=args.cspath_json_file,
                               cpath_dir=args.cpath_dir, spath_dir=args.spath_dir)

    fail = 0
    td = tqdm(iter(dataset))
    for data in td:
        try:
            img = data[0]
            spath = data[1]
            if os.path.exists(spath):
                continue
        except Exception as e:
            fail += 1
            td.set_description(f"Failed:{fail}")
            continue

        output = net(img, is_training=False)
        x_code = output['vq_output']['encoding_indices']

        np.save(file=spath, arr=x_code.asnumpy().flatten())
