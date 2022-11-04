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
""" data_three """

import os
import json
import numpy as np
import mindspore.dataset.vision as C
from mindspore.dataset.vision.utils import Inter
from PIL import Image

from src.config import config

global_ids = {}
class TxtData():
    """ TxtData """
    def __init__(self, db_dir):
        self.db_dir = db_dir
        self.db_dir_json = self.db_dir + "_json"
        with open(f'{db_dir}/meta.json', 'r') as f:
            meta = json.load(f)
        # meta = json.load(open(f'{db_dir}/meta.json', 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

    def __getitem__(self, id_):
        path_json = os.path.join(self.db_dir, id_ + ".json")
        if not os.path.exists(path_json):
            path_json = os.path.join(self.db_dir_json, id_ + ".json")
        txt_dump = json.load(open(path_json))
        if "laion400m" in id_:
            txt_dump["input_ids"] = txt_dump["input_ids"][3:]
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
        if self.use_patch:
            self.patch_size = opts.patch_size
        print(f"image_size => {self.image_size} use_patch => {self.patch_size}")

    def get_image(self, id_):
        """get image

        Args:
            id_ (string): relative path of image

        Returns:
            img: image data
        """
        if ".jpg" in id_:
            num = id_.split(".jpg")[-1]
            id_ = id_.replace(".jpg" + num, ".jpg")
            image_path = os.path.join(self.img_dir, id_)
            if not os.path.exists(image_path):
                id_ = id_.split('/')[-1]
                image_path = os.path.join(self.img_dir, id_)
        elif "cc3m" in id_ or "cc12m" in id_:
            image_path = os.path.join(self.img_dir, id_)
        else:
            image_path = os.path.join(self.img_dir, id_ + ".jpg")

        interpolation = "BILINEAR"
        resize = self.image_size
        image_size = self.image_size

        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

        if hasattr(Inter, interpolation):
            interpolation = getattr(Inter, interpolation)
        else:
            interpolation = Inter.BILINEAR
            print('cannot find interpolation_type: {}, use {} instead'.format(interpolation, 'BILINEAR'))

        trans = [
            C.Resize(resize, interpolation=interpolation),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        for tran in trans:
            image = tran(image)

        return image

    def get_image_patch(self, id_):
        """get image pathches

        Args:
            id_ (string): relative path of image

        Returns:
            numpy: image pathches
        """
        image = self.get_image(id_)
        p = self.patch_size
        channels, h, w = image.shape
        x = np.reshape(image, (channels, h // p, p, w // p, p))
        x = np.transpose(x, (1, 3, 0, 2, 4))
        patches = np.reshape(x, ((h // p) * (w // p), channels * p * p))

        return patches, image

    def get_image_feat(self, id_):
        """get image feats

        Args:
            id_ (string): relative path of image feat

        Returns:
            numpy: image feats
        """
        if ".jpg" in id_:
            num = id_.split(".jpg")[-1]
            feat_path = os.path.join(self.img_dir, id_.replace(".jpg" + num, ".jpg.npz"))
            if not os.path.exists(feat_path):
                feat_path = os.path.join(self.img_dir, id_.replace(".jpg" + num, ".npz"))
        else:
            feat_path = os.path.join(self.img_dir, id_ + ".npz")

        data = np.load(feat_path)

        np_att_feat = data['feat']
        np_pred_boxes = data['pred_boxes']
        np_scores = data['scores']
        np_width = data['width']
        np_height = data['height']

        att_feat = np.array(np_att_feat).astype(np.float32)
        att_feat = att_feat[:config.MAX_IMG_LEN, :]

        box_width = np_pred_boxes[:config.MAX_IMG_LEN, 2] - np_pred_boxes[:config.MAX_IMG_LEN, 0]
        box_height = np_pred_boxes[:config.MAX_IMG_LEN, 3] - np_pred_boxes[:config.MAX_IMG_LEN, 1]
        scaled_width = box_width / np_width
        scaled_height = box_height / np_height
        scaled_x = np_pred_boxes[:config.MAX_IMG_LEN, 0] / np_width
        scaled_y = np_pred_boxes[:config.MAX_IMG_LEN, 1] / np_height

        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]

        pred_boxes = np.concatenate((scaled_x, scaled_y,
                                     scaled_x + scaled_width,
                                     scaled_y + scaled_height,
                                     scaled_width, scaled_height,
                                     scaled_width * scaled_height), axis=1)
        pred_boxes = np.array(pred_boxes).astype(np.float32)

        scores = np.array(np_scores).astype(np.float32)
        scores = scores[:config.MAX_IMG_LEN]

        # pred_classes = np.array(np_pred_classes).astype(np.float32)
        # pred_classes = pred_classes[:config.MAX_IMG_LEN]

        return att_feat, pred_boxes, scores, None

    def __getitem__(self, id_):
        if self.use_patch:
            patches, image = self.get_image_patch(id_)
            return patches, None, None, None, image
        att_feat, pred_boxes, scores, pred_classes = self.get_image_feat(id_)
        return att_feat, pred_boxes, scores, pred_classes, None
class AudioData():
    """ AudioData """

    def __init__(self, audio_dir):
        self.audio_dir = audio_dir

    def __getitem__(self, file_name):

        path_npz = os.path.join(self.audio_dir, file_name + ".npz")
        if "cc3m" in path_npz:                  #special situation for cc3m on server249
            path_npz = path_npz.replace("/training", "").replace("/validation", "")
        feat = np.load(path_npz)['feat']

        # T * 512
        audio_feat = np.array(feat).astype(np.float32)
        if audio_feat.shape[-1] != config.AUDIO_DIM:
            audio_feat = audio_feat.T
        audio_feat = audio_feat[:config.MAX_AUDIO_LEN, :]
        return audio_feat

def get_ids_three(ids_path):
    ids = json.load(open(ids_path))
    size, rank = get_size_rank()
    return ids[rank::size]

def get_size_rank():
    size, rank = 1, 0
    return size, rank

# Image feature, Text token, Audio feature
class TxtImgAudioDataset():
    """ TxtImgAudioDataset """

    def __init__(self, ids_path, txt_db=None, img_db=None, audio_db=None):
        assert txt_db is None or isinstance(txt_db, TxtData)
        assert img_db is None or isinstance(img_db, ImgData)
        assert audio_db is None or isinstance(audio_db, AudioData)

        self.txt_db = txt_db
        self.img_db = img_db
        self.audio_db = audio_db

        self.ids = get_ids_three(ids_path)
        self.total_len = len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        i = i % len(self.ids)
        id_ = self.ids[i]
        example = self.txt_db[id_]
        example['id'] = id_
        example['img_fname'] = id_
        example['audio_fname'] = id_
        return example

    def _get_img_feat(self, id_):
        img_feat, pred_boxes, _, pred_classes, image = self.img_db[id_]
        return img_feat, pred_boxes, pred_classes, image

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

def get_gather_index_three(txt_lens, num_bbs, num_aus, batch_size, max_len, max_bb, out_size):
    assert len(txt_lens) == len(num_bbs) == batch_size
    gather_index = np.expand_dims(np.arange(0, out_size, dtype=np.int64), 0).repeat(batch_size, axis=0)

    for i, (tl, nbb, nau) in enumerate(zip(txt_lens, num_bbs, num_aus)):
        gather_index[i, tl:tl + nbb] = np.arange(max_len, max_len + nbb, dtype=np.int64)
        # 32, 144 - 121
        gather_index[i, tl + nbb:tl + nbb + nau] = np.arange(max_len + max_bb, max_len + max_bb + nau, dtype=np.int64)

    return gather_index

# pad tensors
def pad_tensors(tensors, lens=None, pad=0, max_len=config.MAX_DEFAULT_LEN):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.shape[0] for t in tensors]
    if max_len == -1:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[-1]
    dtype = tensors[0].dtype
    output = np.zeros((bs, max_len, hid), dtype=dtype)
    if pad:
        output.fill(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output[i, :l, ...] = t
    return output

def pad_tensors_pos(tensors, lens, feat, max_len=config.MAX_DEFAULT_LEN):
    """ pad_tensors_pos """
    if tensors is None or tensors[0] is None:
        return np.expand_dims(np.arange(0, feat.shape[1], dtype=np.int64), 0)
    return pad_tensors(tensors, lens, max_len=max_len)

def get_gather_index(txt_lens, num_bbs, batch_size, max_len, out_size):
    """ get_gather_index """

    assert len(txt_lens) == len(num_bbs) == batch_size
    gather_index = np.expand_dims(np.arange(0, out_size, dtype=np.int64), 0).repeat(batch_size, axis=0)

    for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
        gather_index[i, tl:tl + nbb] = np.arange(max_len, max_len + nbb, dtype=np.int64)

    return gather_index

def stack_images(images):
    if images is None or images[0] is None:
        return None
    return np.stack(images)
