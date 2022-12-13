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
import sys
import yaml
from pathlib2 import Path

from mindspore.train.serialization import load_checkpoint


def str2bool(b):
    if b.lower() in ["false"]:
        output = False
    elif b.lower() in ["true"]:
        output = True
    else:
        raise Exception("Invalid Bool Value")
    return output


def parse_with_config(parser):
    """Parse With Config"""
    args = parser.parse_args()
    if args.config is not None:
        config_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args


def download(args, rank_id=0):
    """download ckpt from obs for finetune model."""
    import moxing as mox
    obs_ckpt_path = args.finetune_ckpt
    ckpt_name = obs_ckpt_path.split("/")[-1]
    ckpt_path = args.ckpt_save_dir + str(rank_id)
    local_ckpt_path = os.path.join(ckpt_path, ckpt_name)
    mox.file.copy_parallel(obs_ckpt_path, local_ckpt_path)
    params_dict = None
    if os.path.exists(local_ckpt_path):
        args.logger.info(f"start loading {local_ckpt_path}.")
        params_dict = load_checkpoint(local_ckpt_path)
        args.logger.info(f"end loading {local_ckpt_path}.")
    return params_dict
