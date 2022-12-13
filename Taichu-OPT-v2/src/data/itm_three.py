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
"""itm_three"""

import random
import numpy as np
import math
from toolz.sandbox import unzip

from ..config import config
from .utils import pad_sequence
from .data_three import (stack_images,
                         TxtImgTwoDataset,
                         TxtData, ImgData)

def random_word_fix(tokens, mask, default_num=10):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    total_len = len(tokens)
    mask_len = math.ceil(default_num * 0.15)
    mask_num = random.sample([_ for _ in range(total_len)], mask_len)
    output_label = [-1 for _ in range(total_len)]
    for mask_index in mask_num:
        token = tokens[mask_index]
        tokens[mask_index] = mask
        output_label[mask_index] = token
    return tokens, output_label

def sample_index(total_len, cur_index):
    to_index = cur_index
    while to_index == cur_index:
        to_index = random.randint(0, total_len - 1)
    return to_index

class ItmHardTwoDataset(TxtImgTwoDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """
    def __init__(self, ids_path, txt_db, img_db, use_mask_fix=False):
        super().__init__(ids_path, txt_db, img_db)

        assert isinstance(txt_db, TxtData)
        assert isinstance(img_db, ImgData)

        self.txt_db = txt_db
        self.img_db = img_db
        self.use_mask_fix = use_mask_fix

    def get_random_id(self, i, label):
        """get_random_id"""
        img_index = i
        if label == 0:
            img_index = sample_index(self.total_len, i)
        return img_index

    def __getitem__(self, i):
        example = super().__getitem__(i)

        input_text = example['input_ids'][:config.MAX_TEXT_LEN]

        # text input
        input_ids = self.txt_db.combine_inputs(input_text)

        image, patch_len = self._get_img_feat(example['id'])

        attn_masks = np.ones(len(input_ids) + patch_len, dtype=np.int64)
        attn_masks_text = np.ones(len(input_ids), dtype=np.int64)
        attn_masks_img = np.ones(patch_len, dtype=np.int64)

        input_ids_mask, txt_labels = self.create_mlm_io(input_text)

        target = random.randint(0, 1)
        img_rand_index = self.get_random_id(i, target)
        image_rand, patch_size = self._get_img_feat(self.ids[img_rand_index]['image'])

        return (input_ids, attn_masks_text, input_ids_mask, txt_labels,
                image, attn_masks_img, attn_masks,
                image_rand, target)

    def create_mlm_io(self, input_ids):
        """ create_mlm_io """
        # add mask
        input_ids, txt_labels = random_word_fix(input_ids, self.txt_db.mask)

        # add cls and sep
        input_ids = np.array([self.txt_db.cls_]
                             + input_ids
                             + [self.txt_db.sep])
        txt_labels = np.array([-1] + txt_labels + [-1])
        return input_ids, txt_labels

def itmHardTwo_collate(inputs):
    (tmp_input_ids, attn_masks_text, input_ids_mask, txt_labels,
     images, attn_masks_img, attn_masks,
     images_rand, targets) = map(list, unzip(inputs))

    # text batches
    input_ids = pad_sequence(tmp_input_ids, batch_first=True, padding_value=0, max_lens=config.MAX_FULL_TEXT_LEN)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64), 0)
    input_ids_mask = pad_sequence(input_ids_mask, batch_first=True, padding_value=0, max_lens=config.MAX_FULL_TEXT_LEN)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1, max_lens=config.MAX_FULL_TEXT_LEN)

    # image batches
    images = stack_images(images)
    images_rand = stack_images(images_rand)
    targets = np.array(targets)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=config.MAX_IMG_TEXT_LEN)
    attn_masks_text = pad_sequence(attn_masks_text, batch_first=True, padding_value=0, max_lens=config.MAX_FULL_TEXT_LEN)
    attn_masks_img = pad_sequence(attn_masks_img, batch_first=True, padding_value=0, max_lens=config.MAX_IMG_LEN)

    txt_mask = np.stack((txt_labels != -1).nonzero(), 1)
    txt_label_mask = txt_labels[txt_labels != -1]

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'images': images,
             'attn_masks': attn_masks,
             'attn_masks_text': attn_masks_text,
             'attn_masks_img': attn_masks_img,
             'input_ids_mask': input_ids_mask,
             'txt_mask': txt_mask,
             'txt_label_mask': txt_label_mask,
             'images_rand': images_rand,
             'targets': targets}
    return batch