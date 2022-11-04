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
"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

ITM Dataset
"""
import random
import os
import numpy as np
from toolz.sandbox import unzip

from src.config import config
from src.data.utils import pad_sequence
from src.data.data_three import pad_tensors_pos
from src.data.data_three import (get_gather_index_three, stack_images,
                  TxtImgAudioDataset, get_ids_three,
                  TxtData, ImgData, AudioData)
from src.data.data_three import pad_tensors

def _has_overlap(la, lb):
    if len(la) < len(lb):
        la, lb = lb, la
    s = set(la)
    return any(b in s for b in lb)


def sample_negative(sample_pool, ground_truths, num_sample):
    """ random and retry """
    outputs = ground_truths[:1]
    while _has_overlap(outputs, ground_truths):
        outputs = random.sample(sample_pool, num_sample)
    return outputs


def sample_index(total_len, cur_index):
    to_index = cur_index
    while to_index == cur_index:
        to_index = random.randint(0, total_len - 1)
    return to_index


def sample_index1(total_len, cur_index, cur_index1):
    to_index = cur_index
    while to_index in [cur_index, cur_index1]:
        to_index = random.randint(0, total_len - 1)
    return to_index


class ItmThreeDataset(TxtImgAudioDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """

    def __init__(self, ids_path, txt_db, img_db, audio_db, neg_sample_p=0.5, use_mask_fix=False):
        super().__init__(ids_path, txt_db, img_db, audio_db)
        assert isinstance(txt_db, TxtData)
        assert isinstance(img_db, ImgData)
        assert isinstance(audio_db, AudioData)

        self.txt_db = txt_db
        self.img_db = img_db
        self.audio_db = audio_db
        self.use_mask_fix = use_mask_fix

        self.ids = get_ids_three(ids_path)
        self.total_len = len(self.ids)
        rank_id_str = os.getenv('RANK_ID', '0')
        rank_id = int(rank_id_str[rank_id_str.rfind('-') + 1:])  # 'RANK_ID': 'job24535502-job-facereidtome-hn-0/1'
        # self.path_lens = self.path_lens.split('.')[0] + '_itm' + str(rank_id) + '.json'
        # self.path_lens = self.path_lens + "_itm.json"

        print("itm ids {}".format(self.total_len))

        self.neg_sample_p = neg_sample_p

    def get_random_id(self, i, id_, label):
        """get_random_id"""
        img_id = id_
        audio_id = id_
        if label == 0:
            to_index = sample_index(len(self.ids), i)
            img_id = self.ids[to_index]
            audio_id = self.ids[to_index]
        elif label == 1:
            # img_id = sample_negative(self.all_imgs, [img_id], 1)[0]
            to_index = sample_index(len(self.ids), i)
            img_id = self.ids[to_index]
        elif label == 2:
            # audio_id = sample_negative(self.all_audios, [audio_id], 1)[0]
            to_index = sample_index(len(self.ids), i)
            audio_id = self.ids[to_index]
        elif label == 3:
            # img_id = sample_negative(self.all_imgs, [img_id], 1)[0]
            # audio_id = sample_negative(self.all_audios, [audio_id], 1)[0]
            to_index = sample_index(len(self.ids), i)
            to_index1 = sample_index1(len(self.ids), i, to_index)
            img_id = self.ids[to_index]
            audio_id = self.ids[to_index1]
        return img_id, audio_id

    def __getitem__(self, i):
        example = super().__getitem__(i)
        # labels and negative images should be sampled every epoch
        ground_truth_label = random.randint(0, 4)

        # text input
        input_ids = example['input_ids']
        if self.use_mask_fix:
            input_ids = input_ids[:config.MAX_TEXT_LEN]
        input_ids = self.txt_db.combine_inputs(input_ids)

        img_id, audio_id = self.get_random_id(i, self.ids[i], ground_truth_label)
        # image
        img_feat, img_pos_feat, _, image = self._get_img_feat(img_id)

        # audio
        audio_feat, _ = self._get_audio_feat(audio_id)

        attn_masks = np.ones(len(input_ids) + img_feat.shape[0] + audio_feat.shape[0], dtype=np.int64)

        target = ground_truth_label

        return input_ids, img_feat, img_pos_feat, attn_masks, target, audio_feat, image



def itmThree_collate(inputs):
    """ itmThree_collate """

    (input_ids, img_feats, img_pos_feats, attn_masks, targets, audio_feats, images
     ) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.shape[0] for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64
                                            ), 0)

    # image batches
    num_bbs = [f.shape[0] for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs, max_len=config.MAX_IMG_LEN)
    img_pos_feat = pad_tensors_pos(img_pos_feats, num_bbs, img_feat, max_len=config.MAX_IMG_LEN)
    images = stack_images(images)

    # audio batches
    num_aus = [f.shape[0] for f in audio_feats]
    audio_feat = pad_tensors(audio_feats)
    audio_pos_ids = np.expand_dims(np.arange(0, audio_feat.shape[1], dtype=np.int64
                                             ), 0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=config.MAX_FULL_LEN)
    # targets = np.concatenate(targets, axis=0)
    targets = np.array(targets)

    bs, max_tl = input_ids.shape
    max_bb = img_feat.shape[1]
    out_size = attn_masks.shape[1]

    gather_index = get_gather_index_three(txt_lens, num_bbs, num_aus, bs, max_tl, max_bb, out_size)
    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets,
             'audio_feat': audio_feat,
             'audio_pos_ids': audio_pos_ids,
             'images': images}
    return batch
