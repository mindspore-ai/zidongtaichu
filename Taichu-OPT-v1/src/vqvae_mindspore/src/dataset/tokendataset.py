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
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.common.dtype as mstype
from mindspore.communication import get_rank, get_group_size

class DatasetGenerator:
    def __init__(self, img_size, cspath_json, cpath_dir=None, spath_dir=None):
        """
        Loading image files as a dataset generator
        Args:
            ids_path: path of the json file that record image files as [path_img1, path_img2,...]
            data_dir: the root dir of [path_img1, path_img2,...].
                      The absolute path of img1 should be: os.path.join(data_dir, path_img1)
        """
        self._index = 0
        self._imgsize = img_size

        print(cspath_json)
        cspath = json.load(open(cspath_json, 'r'))
        self.cpath, self.spath = [], []
        for (cpath, spath) in cspath:
            self.cpath.append(cpath)
            self.spath.append(spath)
        if cpath_dir is not None:
            self.cpath = [os.path.join(cpath_dir, f) for f in self.cpath]
        if spath_dir is not None:
            self.spath = [os.path.join(spath_dir, f) for f in self.spath]

        self._rescale = 1.0 / 255.0
        self._shift = 0.0

    def __getitem__(self, index):
        try:
            cur_img = Image.open(self.cpath[index]).convert('RGB')
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
                right = (width + height) // 2 + move

            cur_img = cur_img.crop((left, top, right, bottom))
            cur_img = cur_img.resize((self._imgsize, self._imgsize), Image.BILINEAR)
            image = np.array(cur_img).astype(np.float32)
            image = np.expand_dims(image, axis=0)
            # Rescale
            image = (image - self._shift) * self._rescale
            # BHWC - BCHQ
            image = np.transpose(image, [0, 3, 1, 2])
            image = ms.Tensor(image, dtype=mstype.float32)

            return image, self.spath[index]
        except Exception as e:
            print("File {} Error {}".format(self.cpath[index], e))
            return self.__next__()

    def __next__(self):
        if self._index >= len(self.spath) - 1:
            raise StopIteration
        else:
            self._index += 1
            return self.__getitem__(self._index)

    def __iter__(self):
        self._index = 0
        return self

    def __len__(self):
        return len(self.cpath)


def create_dataset_for_token(cspath_json, cpath_dir, spath_dir,
                             img_size, batch_size, buffer_size,
                             ifdist, num_workers):
    """
    create dataset for train or test
    """
    dataset = DatasetGenerator(img_size=img_size,
                               cspath_json=cspath_json,
                               cpath_dir=cpath_dir,
                               spath_dir=spath_dir)
    return dataset