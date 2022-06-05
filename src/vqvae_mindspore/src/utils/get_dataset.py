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
try:
    from src.utils.get_logger import get_logger_dist as get_logger
    from src.dataset.cifar10 import create_dataset_cifar10
    from src.dataset.customdataset import create_dataset_custom
    from src.dataset.imagenet import create_dataset_imagenet
    from src.dataset.tokendataset import create_dataset_for_token
except:
    from src.vqvae_mindspore.src.utils.get_logger import get_logger_dist as get_logger
    from src.vqvae_mindspore.src.dataset.cifar10 import create_dataset_cifar10
    from src.vqvae_mindspore.src.dataset.customdataset import create_dataset_custom
    from src.vqvae_mindspore.src.dataset.imagenet import create_dataset_imagenet
    from src.vqvae_mindspore.src.dataset.tokendataset import create_dataset_for_token

def create_datasets(data_opt, dist=False):
    logger = get_logger()
    name = data_opt['name'].lower()

    if name == 'cifar10':
        trainset = create_dataset_cifar10(ifdist=dist,
                                          data_path=data_opt['train_path'],
                                          img_size=data_opt['img_size'],
                                          batch_size=data_opt['batchsize'],
                                          buffer_size=data_opt['buffersize'],
                                          repeat_size=data_opt['repeatsize'])
        validset = create_dataset_cifar10(ifdist=dist,
                                          data_path=data_opt['valid_path'],
                                          img_size=data_opt['img_size'],
                                          batch_size=data_opt['batchsize'],
                                          buffer_size=data_opt['buffersize'],
                                          repeat_size=data_opt['repeatsize'])
        return trainset, validset

    elif name == 'imagenet':
        logger.info(f"Loading ImageNet datasets...")
        trainset = create_dataset_imagenet(ifdist=dist,
                                          data_path=data_opt['train_path'],
                                          img_size=data_opt['img_size'],
                                          batch_size=data_opt['batchsize'],
                                          buffer_size=data_opt['buffersize'],
                                          repeat_size=data_opt['repeatsize'],
                                          num_parallel_workers=data_opt['num_workers'])
        validset = create_dataset_imagenet(ifdist=dist,
                                          data_path=data_opt['valid_path'],
                                          img_size=data_opt['img_size'],
                                          batch_size=data_opt['batchsize'],
                                          buffer_size=data_opt['buffersize'],
                                          repeat_size=data_opt['repeatsize'],
                                          num_parallel_workers=data_opt['num_workers'])
        return trainset, validset

    else:
        logger.info(f"Loading {name} datasets through Cunstom method...")
        trainset = create_dataset_custom(ifdist=dist,
                                         data_dir=data_opt['data_dir'],
                                         ids_path=data_opt['train_ids_path'],
                                         img_size=data_opt['img_size'],
                                         batch_size=data_opt['batchsize'],
                                         buffer_size=data_opt['buffersize'],
                                         repeat_size=data_opt['repeatsize'],
                                         num_workers=data_opt['num_workers'])
        validset = create_dataset_custom(ifdist=dist,
                                         ids_path=data_opt['valid_ids_path'],
                                         img_size=data_opt['img_size'],
                                         data_dir=data_opt['data_dir'],
                                         batch_size=data_opt['batchsize'],
                                         buffer_size=data_opt['buffersize'],
                                         repeat_size=data_opt['repeatsize'],
                                         num_workers=data_opt['num_workers'])
        return trainset, validset

def get_dataset(data_opt):
    return create_datasets(data_opt=data_opt, dist=False)


def get_dataset_dist(data_opt):
    return create_datasets(data_opt=data_opt, dist=True)

def get_TokenDataset(data_opt, cspath_json, cpath_dir, spath_dir, dist=False):
    dataset = create_dataset_for_token(ifdist=dist,
                                       cspath_json=cspath_json,
                                       cpath_dir=cpath_dir,
                                       spath_dir=spath_dir,
                                       img_size=data_opt['img_size'],
                                       batch_size=data_opt['batchsize'],
                                       buffer_size=data_opt['buffersize'],
                                       num_workers=data_opt['num_workers'])
    return dataset