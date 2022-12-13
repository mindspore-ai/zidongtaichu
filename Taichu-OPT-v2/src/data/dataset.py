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
Create dataset for training and evaluating
"""
from mindspore.dataset import GeneratorDataset
from src.data import MetaLoaderTwo, data_column_two
from src.tools.misc import set_random_seed
from src.data.pretrain_three_data import create_three_dataloaders

def create_dataset(opts, device_num=1, is_train=True):
    """
    Create dataset

    Inputs:
        opts: config file which including dataset path
        device_num: total device number
        rank: current rank id
        column_name: the column name of the train file. Default is a list
        batch_size: batch size
        full_batch: whether do full batch operation.
        drop_remainder: whether drop remainder

    Returns:
        dataset_restore: the dataset for training
    """

    if isinstance(opts.ids_train_path, list):
        opts.ids_train_path = ",".join(opts.ids_train_path)

    set_random_seed(opts.seed)
    if is_train:
        train_data_loaders, datalen = create_three_dataloaders(opts.ids_train_path, opts.train_datasets, is_train,
                                                      opts, device_num=device_num)
        batch_size = opts.train_batch_size
    else:
        train_data_loaders, datalen = create_three_dataloaders(opts.ids_val_path, opts.val_datasets, is_train,
                                                      opts, device_num=device_num)
        batch_size = opts.val_batch_size

    datalen = datalen // batch_size

    metaloader = MetaLoaderTwo(train_data_loaders, datalen=datalen, task_num=len(train_data_loaders.keys()))
    dataset = GeneratorDataset(metaloader, column_names=data_column_two, shuffle=False)

    # If eod_reset enabled, another two inputs will be generated through input_ids
    return dataset
