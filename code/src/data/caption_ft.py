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
""" caption_ft """

import numpy as np
from toolz.sandbox import unzip
from .data_three import pad_tensors, pad_tensors_pos
from .data_three import TxtImgTwoDataset
from .utils import pad_sequence
from src.config import config


class CaptionDataset(TxtImgTwoDataset):
    """ CaptionDataset """

    def __init__(self, ids_path, txt_db, img_db):
        super().__init__(ids_path, txt_db, img_db)

    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        ids = example['id']

        txt_inputs, txt_gts, txt_masks = self._get_txt_token(example['input_ids'])

        # text mask input and gt text
        # input_ids, txt_labels = self.create_mlm_io(example['input_ids'])

        # img input

        img_feat, _, _,image = self._get_img_feat(example['img_fname'])
        img_pos_feat  = np.arange(img_feat.shape[0] +1)
        attn_masks = np.ones(img_pos_feat.shape[0], dtype=np.int64)


        return (ids, img_feat, img_pos_feat, attn_masks,
                txt_inputs, txt_gts, txt_masks)


def caption_collate(inputs):

    (ids, img_feats, img_pos_feats, attn_masks,
     txt_inputs, txt_gts, txt_masks) = map(list, unzip(inputs))

    # image batches
    num_bbs = [f.shape[0] for f in img_feats]
    if len(img_feats[0].shape) == 3:
        ### 3-dimension raw patch:
        img_feat = np.stack(img_feats,axis=0)
    else:
        ### 2-dimension feat
        img_feat = pad_tensors(img_feats, num_bbs, max_len=config.MAX_IMG_LEN)
    if img_pos_feats[0] is not None:
        if len(img_pos_feats[0].shape) == 1: ### for vit patch
            img_pos_feat = np.stack(img_pos_feats,axis=0)
        else:
            
            img_pos_feat = pad_tensors_pos(img_pos_feats, num_bbs, img_feat, max_len=config.MAX_IMG_LEN)
    else:
        img_pos_feat = None ### for vit feature

    # audio batches

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=config.MAX_IMG_LEN)

    out_size = attn_masks.shape[1]
    batch_size = attn_masks.shape[0]
    gather_index = np.expand_dims(np.arange(0, out_size, dtype=np.int64), 0).repeat(batch_size, axis=0)

    # txt decoder
    txt_inputs = pad_sequence(txt_inputs, batch_first=True, padding_value=0, max_lens=config.MAX_TEXT_LEN)
    txt_gts = pad_sequence(txt_gts, batch_first=True, padding_value=0, max_lens=config.MAX_TEXT_LEN)
    txt_masks = pad_sequence(txt_masks, batch_first=True, padding_value=0, max_lens=config.MAX_TEXT_LEN)


    batch = {'input_ids': None,
             'position_ids': None,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': None,
             'audio_feat': None,
             'audio_pos_ids': None,
             'txt_inputs': txt_inputs,
             'txt_gts': txt_gts,
             'txt_masks': txt_masks,
             'txt_pos': None,
             'ids': ids,
             'img_masks': None,
             'audio_masks': None,
             'mask_types': None}
    return batch