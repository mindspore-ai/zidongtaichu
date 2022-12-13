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

from src.data import (BatchSampler, EasySampler,
                  TxtData, ImgData, AudioData,
                  ItmHardTwoDataset, itmHardTwo_collate)

from src.data.data_loader import DataLoader
from src.tools.logger import LOGGER

def build_naive_dataloader(dataset, collate_fn, batch_size, device_num):
    sampler = BatchSampler(len(dataset), batch_size=batch_size, device_num=device_num)
    loader = DataLoader(dataset, batch_sampler=sampler,collate_fn=collate_fn, device_num=device_num, is_train=False)
    return loader

def build_dataloader_ms(dataset, collate_fn, is_train, opts, device_num):
    if is_train:
        batch_size = opts.train_batch_size
    else:
        batch_size = opts.val_batch_size

    sampler = EasySampler(dataset, batch_size=batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, device_num=device_num, full_batch=opts.full_batch)

    return loader

def build_itmHardTwo_dataset(ids_path, txt_db, img_db, opts):
    dataset = ItmHardTwoDataset(ids_path, txt_db, img_db)
    return dataset, itmHardTwo_collate


def create_three_dataloaders(ids_path, datasets, is_train, opts, device_num, ids_two_path=None,
                             ids_textaudio_path=None):
    """ Create dataloaders """
    dataloaders = {}

    ## pretrain tasks
    for dset in datasets:
        if dset['tasks']:  # if the list sequence is empty, then it is equal to False
            txt_db = TxtData(dset['db'][0])
            img_db = ImgData(dset['img'][0], opts)
            audio_db = AudioData(dset['audio'][0])

        for i, t in enumerate(dset['tasks']):
            task = f'{t}_{dset["name"]}'
            if task.startswith('itmTwo'):
                dataset, collate_fn = build_itmHardTwo_dataset(ids_path, txt_db, img_db, opts)
            else:
                raise ValueError('Undefined task %s'% (task))
            LOGGER.info("Create Dataset %s Success", (task))
            loader = build_dataloader_ms(dataset, collate_fn, is_train, opts, device_num)

            if is_train:
                ratio = dset['mix_ratio'][i]
                dataloaders[task] = (loader, ratio)
            else:
                dataloaders[task] = loader
    return dataloaders, len(dataset)
