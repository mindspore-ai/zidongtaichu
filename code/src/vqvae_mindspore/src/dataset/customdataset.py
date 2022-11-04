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
from PIL import Image
import numpy as np
# import ipdb
from random import shuffle
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
from mindspore.communication import get_rank, get_group_size

class DatasetGenerator:
    def __init__(self, ids_path, data_dir=None, replicate = 20):
        """
        Loading image files as a dataset generator
        Args:
            ids_path: path of the json file that record image files as [path_img1, path_img2,...]
            data_dir: the root dir of [path_img1, path_img2,...].
                      The absolute path of img1 should be: os.path.join(data_dir, path_img1)
        """
        data = json.load(open(ids_path, 'r'))
        if data_dir is not None:
            data = [os.path.join(data_dir, item) for item in data]
        self.data = []
        for i in range(replicate):
            shuffle(data)
            self.data += data

    def __next__(self, index):
        if index >= self.__len__() - 1:
            return self.__getitem__(0)
        else:
            return self.__getitem__(index + 1)

    def __getitem__(self, index):
        try:
            cur_img = Image.open(self.data[index]).convert('RGB')
            width, height = cur_img.size
            if width % 2 == 1:
                width -= 1
            if height % 2 == 1:
                height -= 1
            # random crop
            if width == height:
                cur_img = np.array(cur_img).astype(np.float32)
                return np.expand_dims(cur_img, axis=0)
            elif width < height:
                diff = height - width
                move = np.random.choice(diff) - diff // 2
                left, right = 0, width
                top = (height - width) // 2 + move
                bottom = (height + width) // 2 + move
            else:
                diff = width - height
                move = np.random.choice(diff) - diff // 2
                top, bottom = 0, height
                left = (width - height) // 2 + move
                right = (width + height) //2 + move

            cur_img = cur_img.crop((left, top, right, bottom))
            image = np.array(cur_img).astype(np.float32)
            image = np.expand_dims(image, axis=0)
            # ret = np.concatenate([image, image], axis=0)
            # return image, image
            return image
        except Exception as e:
            print("File Error {}".format(e))
            return self.__next__(index+1)

    def __len__(self):
        return len(self.data)


def create_dataset_custom(ids_path, data_dir, img_size, batch_size,
                          buffer_size, repeat_size, ifdist, num_workers=4):
    """
    create dataset for train or test
    """
    dataset = DatasetGenerator(ids_path=ids_path, data_dir=data_dir)
    if ifdist is False:
        custom_ds = ds.GeneratorDataset(dataset, ["image"], shuffle=True,
        # custom_ds = ds.GeneratorDataset(dataset, ["image", "label"], shuffle=True,
                                        num_parallel_workers=num_workers)
    else:
        custom_ds = ds.GeneratorDataset(dataset, ["image"], shuffle=True,
        # custom_ds = ds.GeneratorDataset(dataset, ["image", "label"], shuffle=True,
                                        num_parallel_workers=num_workers,
                                        num_shards=get_group_size(), shard_id=get_rank())

    rescale = 1.0 / 255.0
    shift = 0.0
    transforms = [
        CV.Resize((img_size, img_size)),
        CV.Rescale(rescale, shift),
        # CV.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        CV.HWC2CHW()
    ]

    custom_ds = custom_ds.map(input_columns="image", num_parallel_workers=num_workers, operations=transforms)
    # custom_ds = custom_ds.map(input_columns="label", num_parallel_workers=num_workers, operations=transforms)
    custom_ds = custom_ds.shuffle(buffer_size=buffer_size)
    custom_ds = custom_ds.batch(batch_size, drop_remainder=True)
    custom_ds = custom_ds.repeat(repeat_size)
    return custom_ds