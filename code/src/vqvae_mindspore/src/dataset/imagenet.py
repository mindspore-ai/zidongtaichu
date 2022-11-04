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

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
from mindspore.communication.management import get_rank, get_group_size

def create_dataset_imagenet(data_path, img_size, batch_size, buffer_size,
                            repeat_size, ifdist, num_parallel_workers=2):
    """
    create dataset for train or test
    """
    if ifdist is False:
        cifar_ds = ds.ImageFolderDataset(data_path, num_parallel_workers=num_parallel_workers, shuffle=True)
    else:
        cifar_ds = ds.ImageFolderDataset(data_path, shuffle=True, num_parallel_workers=num_parallel_workers,
                                         shard_id=get_rank(), num_shards=get_group_size())

    rescale = 1.0 / 255.0
    shift = 0.0

    transforms = [
        CV.Decode(),
        CV.Resize((img_size, img_size)),
        CV.Rescale(rescale, shift),
        # CV.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        CV.HWC2CHW()
    ]

    cifar_ds = cifar_ds.map(input_columns="image",
                            num_parallel_workers=num_parallel_workers,
                            operations=transforms)

    cifar_ds = cifar_ds.shuffle(buffer_size=buffer_size)
    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=True)
    cifar_ds = cifar_ds.repeat(repeat_size)
    return cifar_ds