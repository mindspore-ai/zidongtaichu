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
"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc utilities
"""
import json
import os
import sys
import random
import numpy as np
import mindspore as ms

class NoOp:
    """ useful for distributed training No-Ops """

    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


def parse_with_config(parser):
    """Parse With Config"""
    args = parser.parse_args()
    data_path_dir = args.data_path
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
        
        if args.config.split("/")[-2] == "caption":
            args.caption_eval_gt = os.path.join(data_path_dir, args.caption_eval_gt)
        if args.config.split("/")[-2] == "vqa":
            args.vqa_eval_gt = os.path.join(data_path_dir, args.vqa_eval_gt) 
    del args.config
    if args.model_config is not None:
        args.model_config = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                      "../../..")), args.model_config)
    args.vocab_path = os.path.join(data_path_dir, args.vocab_path)
    if args.ids_train_path is not None:
        args.ids_train_path = os.path.join(data_path_dir, args.ids_train_path)
    if args.ids_val_path is not None:
        args.ids_val_path = os.path.join(data_path_dir, args.ids_val_path)
    train_db_list =[]
    train_img_list = []
    train_db_list.append(os.path.join(data_path_dir, (args.train_datasets[0])['db'][0]))
    train_img_list.append(os.path.join(data_path_dir, (args.train_datasets[0])['img'][0]))
    (args.train_datasets[0])['db'] = train_db_list
    (args.train_datasets[0])['img'] = train_img_list
    val_db_list = []
    val_img_list = []
    val_db_list.append(os.path.join(data_path_dir, (args.val_datasets[0])['db'][0]))
    val_img_list.append(os.path.join(data_path_dir, (args.val_datasets[0])['img'][0]))
    (args.val_datasets[0])['db'] = val_db_list
    (args.val_datasets[0])['img'] = val_img_list
    return args


def set_random_seed(seed):
    """Set Random Seed"""
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)

class Struct:
    def __init__(self, dict_):
        self.__dict__.update(dict_)
