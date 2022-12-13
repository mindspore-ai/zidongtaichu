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
TextToImage Datasets
"""
import os
import numpy as np
from toolz.sandbox import unzip
from src.data.data_three import pad_tensors, pad_tensors_pos
from src.data.data_three import get_ids_three, get_size_rank
from src.data.utils import pad_sequence

# Image feature, Text token, Audio feature
class T2IDetectFeatTxtTokTwoDataset():
    """ DetectFeatTxtTokTwoDataset """

    def __init__(self, ids_path, img_token_path, max_txt_len, txt_db, mode, dataname, batch_size):
        self.txt_db = txt_db
        self.max_txt_len = max_txt_len
        self.img_token_path = img_token_path
        self.mode = mode
        self.dataname = dataname.lower()
        assert self.dataname in ['cc3m', 'coco']

        self._img_feat = np.random.randn(50, 2048).astype(np.float32)
        self._img_pos_feat = np.random.randn(50, 7).astype(np.float32)
        ids = get_ids_three(ids_path)
        length = int((len(ids) // batch_size) * batch_size)
        self.ids = ids[:length]
        size, rank = get_size_rank()
        print("Total data for {} in rank-{}/size-{} is {}.".format(mode, rank, size, len(self.ids)))


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]
        example['id'] = id_
        example['img_fname'] = id_

        # get txt input tokens
        ids = example['id']
        input_ids = np.array(example['input_ids'])
        input_ids = input_ids[:self.max_txt_len]
        attn_masks = np.ones(input_ids.shape[0], dtype=np.int64)

        # get img decode tokens
        if self.dataname == 'cc3m':
            img_id = id_.split('.')[0]
            npy_path = os.path.join(self.img_token_path, img_id + '.npy')
        elif self.dataname == 'coco':
            img_id = id_.split('.')[0].split('_')[-1]
            if 'train' in id_:
                npy_path = os.path.join(self.img_token_path, 'train2014', 'COCO_train2014_' + img_id + '.npy')
            else:
                npy_path = os.path.join(self.img_token_path, 'val2014', 'COCO_val2014_' + img_id + '.npy')

        img_ids = np.load(npy_path).flatten() + 1
        img_inputs, img_gts, img_masks = self._get_img_token(list(img_ids))

        # Unused img features
        # img_feat, img_pos_feat, _ = self._get_img_feat(example['img_fname'])
        img_feat = self._img_feat
        img_pos_feat = self._img_pos_feat

        return (ids, img_feat, img_pos_feat, input_ids, attn_masks,
                img_inputs, img_gts, img_masks)

    def _get_img_token(self, img_ids):
        img_inputs = np.array([0] + img_ids)
        img_gts = np.array(img_ids + [0])
        img_masks = np.ones(len(img_gts))
        return img_inputs, img_gts, img_masks


def t2i_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    :audio_feat   (n, audio_size, audio_dim)
    """
    # Note that here txt_inputs/gts/masks are actually img tokens
    (ids, img_feats, img_pos_feats, input_ids, attn_masks,
     img_inputs, img_gts, img_masks) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.shape[0] for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64), 0)

    # image batches
    num_bbs = [f.shape[0] for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs, max_len=-1)
    if img_pos_feats[0] is not None:
        img_pos_feat = pad_tensors_pos(img_pos_feats, num_bbs, img_feat, max_len=-1)
    else:
        img_pos_feat = None  ### for vit feature

    # attn batches
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=50)

    out_size = attn_masks.shape[1]
    batch_size = attn_masks.shape[0]
    gather_index = np.expand_dims(np.arange(0, out_size, dtype=np.int64), 0).repeat(batch_size, axis=0)

    # txt decoder
    img_inputs = pad_sequence(img_inputs, batch_first=True, padding_value=0, max_lens=-1)
    img_gts = pad_sequence(img_gts, batch_first=True, padding_value=0, max_lens=-1)
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0, max_lens=-1)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': None,
             'audio_feat': None,
             'audio_pos_ids': None,
             'txt_inputs': img_inputs,
             'txt_gts': img_gts,
             'txt_masks': img_masks,
             'txt_pos': None,
             'ids': ids,
             'img_masks': None,
             'audio_masks': None,
             'mask_types': None}
    return batch
