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
""" data_three """

import os
import json
from ..config import config
import numpy as np
import mindspore.dataset.vision.c_transforms as C
from mindspore.dataset.vision.utils import Inter
from PIL import Image
import random
from mindspore.communication.management import get_group_size, get_rank
from .randaugment import RandomAugment

class TxtData():
    """ TxtData """

    def __init__(self, db_dir):

        self.db_dir = db_dir
        self.db_dir_json = self.db_dir + "_json"
        with open(f'{db_dir}/meta.json', 'r') as f:
            meta = json.load(f)
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

    def __getitem__(self, id_):

        path_json = os.path.join(self.db_dir_json, id_ + ".json")
        txt_dump = json.load(open(path_json))
        return txt_dump

    def combine_inputs(self, inputs):
        input_ids = [self.cls_] + inputs[:config.MAX_TEXT_LEN] + [self.sep]
        return np.array(input_ids)


class ImgData():
    """ ImgData """

    def __init__(self, img_dir, opts):
        self.img_dir = img_dir
        self.use_patch = opts.use_patch
        self.image_size = getattr(opts, "image_size", 224)
        self.patch_size = opts.patch_size

        print(f"image_size => {self.image_size} use_patch => {self.patch_size}")

    def get_image(self, id_):

        image_path = os.path.join(self.img_dir, id_)

        resize = self.image_size

        mean = [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
        std = [0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]

        trans = [
            C.RandomResizedCrop(resize, scale=(0.2, 1.0), interpolation=Inter.BICUBIC),
            C.RandomHorizontalFlip(),
            RandomAugment(2, 7, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        for tran in trans:
            image = tran(image)

        return image

    def __getitem__(self, id_):
        image = self.get_image(id_)
        path_len = 1 + (self.image_size//self.patch_size)**2
        return image, path_len

class AudioData():
    """ AudioData """

    def __init__(self, audio_dir):
        self.audio_dir = audio_dir

    def __getitem__(self, file_name):

        path_npz = os.path.join(self.audio_dir, file_name + ".npz")
        if "cc3m" in path_npz:
            path_npz = path_npz.replace("/training", "").replace("/validation", "")
        feat = np.load(path_npz)['feat']

        # T * 512
        audio_feat = np.array(feat).astype(np.float32)
        if audio_feat.shape[-1] != config.AUDIO_DIM:
            audio_feat = audio_feat.T
        audio_feat = audio_feat[:config.MAX_AUDIO_LEN, :]
        return audio_feat


def get_ids_three(ids_path):

    size, rank = get_size_rank()
    ids = []
    for id_path in ids_path.split(","):
        print(f"data load rank: {rank} size: {size} path: {id_path}")
        ids_tmp = json.load(open(id_path))
        ids.extend(ids_tmp)

    ids_rank = ids[rank::size]

    if config.USE_LARGE_DATA:

        cc12m = load_large_data("/mnt/sfs_turbo/baidu_data_1000w_zh/json_cc12m_token/json_cc12m_token_rank_", 256, rank, size)
        ids_rank.extend(cc12m)

        zero = load_large_data("/mnt/sfs_turbo/baidu_data_1000w_zh/json_zero_token/json_zero_train_zh_token_", 256, rank, size)
        ids_rank.extend(zero)

        wukong = load_large_data("/mnt/sfs_turbo/baidu_data_1000w_zh/json_wukong_token/json_wukong_token_", 256, rank, size)
        ids_rank.extend(wukong)

    return ids_rank


def load_large_data(path_root, total, rank, size):
    # path_cc12m = "/mnt/sfs_turbo/baidu_data_1000w_zh/json_cc12m/json_cc12m_token_rank_"
    # path_zero = "/mnt/sfs_turbo/baidu_data_1000w_zh/json_zero/json_zero_train_zh_token_rank_"
    # path_wukong = "/mnt/sfs_turbo/baidu_data_1000w_zh/json_wukong/json_wukong_"

    json_data = []
    num = total // size
    for index in range(rank * num, (rank + 1) * num):
        print(f"data load rank: {rank} size: {size} path: {path_root}{index}.json")
        json_temp = json.load(open(path_root + f"{index}.json"))
        json_data.extend(json_temp)
    return json_data


def get_size_rank():
    size, rank = get_group_size(), get_rank()
    return size, rank

# Image feature, Text token, Audio feature
class TxtImgAudioDataset():
    """ TxtImgAudioDataset """

    def __init__(self, ids_path, txt_db = None, img_db = None, audio_db = None):
        assert txt_db is None or isinstance(txt_db, TxtData)
        assert img_db is None or isinstance(img_db, ImgData)
        assert audio_db is None or isinstance(audio_db, AudioData)

        self.txt_db = txt_db
        self.img_db = img_db
        self.audio_db = audio_db

        self.ids = get_ids_three(ids_path)
        self.total_len = len(self.ids)
        print("Data ids {}".format(self.total_len))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        i = i % len(self.ids)
        anno = self.ids[i]

        example = {}

        img_id = anno['image']
        caption = anno['caption']

        if isinstance(caption[0], list):
            caption = random.choice(caption)

        example['input_ids'] = caption
        example['id'] = img_id

        return example

    def _get_img_feat(self, id_):
        image = self.img_db[id_]
        return image

    def _get_audio_feat(self, id_):
        audio_feat = self.audio_db[id_]
        return audio_feat, audio_feat.shape[0]

    def _get_txt_token(self, input_ids):
        input_ids = input_ids[: config.MAX_TEXT_GTS_LEN]
        txt_inputs = np.array([0] + input_ids)
        txt_gts = np.array(input_ids + [0])
        txt_masks = np.ones(len(txt_gts))
        return txt_inputs, txt_gts, txt_masks


# Image feature, Text token, Audio feature
class TxtImgTwoDataset(TxtImgAudioDataset):
    def __init__(self, ids_path, txt_db, img_db):
        super(TxtImgTwoDataset, self).__init__(ids_path, txt_db=txt_db, img_db=img_db)

class TxtAudioDataset(TxtImgAudioDataset):
    def __init__(self, ids_path, txt_db, audio_db):
        super(TxtAudioDataset, self).__init__(ids_path, txt_db=txt_db, audio_db=audio_db)

class TxtDataset(TxtImgAudioDataset):
    def __init__(self, ids_path, txt_db):
        super(TxtDataset, self).__init__(ids_path, txt_db=txt_db)

def stack_images(images):
    if images is None or images[0] is None:
        return None
    return np.stack(images)

def get_image_from_path(image_path, image_size, patch_size):

    mean = [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
    std = [0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]

    trans = [
        C.Resize(image_size, interpolation=Inter.BILINEAR),
        C.CenterCrop(image_size),
        C.Normalize(mean=mean, std=std),
        C.HWC2CHW()
    ]

    image = Image.open(image_path).convert('RGB')
    image = np.array(image)

    for tran in trans:
        image = tran(image)

    patch_len = 1 + (image_size//patch_size)**2

    return image, patch_len
