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
VQA Datasets
"""
import json

import numpy as np
from toolz.sandbox import unzip

from src.config import config
from src.data.utils import pad_sequence
from src.data.data_three import pad_tensors, pad_tensors_pos, get_gather_index, stack_images
from src.data.data_three import ImgData, TxtImgAudioDataset

class VQATxtData():
    """ TxtTokThreeLmdb """

    def __init__(self, db_dir, name="FM-IQA.json", mode = "train"):

        print("VQATxtTokThreeLmdb {}".format(db_dir))
        with open(f'{db_dir}/{name}') as f:
            data = json.load(f)

        if mode =="train":
            self.data = data[mode]
        elif mode=="val":
            self.data = data[mode][0:75200]

        self.ids = []
        for i in range(len(self.data)):
            self.ids.append(i)

        self.db_dir = db_dir

        if mode=="train":
            with open(f'{db_dir}/train_token_ids.json','r') as f:
                self.token_ids = json.load(f)
        else:
            with open(f'{db_dir}/val_token_ids.json','r') as f:
                self.token_ids = json.load(f)

        with open(f'{db_dir}/meta.json', 'r') as f:
            meta = json.load(f)
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']

    def __getitem__(self, id_):
        txt_dump = {}
        txt_dump['question_id'] = self.data[id_]['question_id']
        txt_dump['image_id'] = self.data[id_]['image_id']

        input_ids = self.token_ids[txt_dump['question_id']]["input_ids"]
        txt_dump['input_ids'] = input_ids

        answer_ids = self.token_ids[txt_dump['question_id']]["answer_ids"]
        txt_dump['answer_ids'] = answer_ids

        return txt_dump

class VQATxtImgDataset(TxtImgAudioDataset):
    """ VQATxtImgDataset """

    def __init__(self, txt_db, img_db, mode):

        assert isinstance(txt_db,VQATxtData)
        assert isinstance(img_db, ImgData)

        self.txt_db = txt_db
        self.img_db = img_db
        self.mode = mode

        self.ids = txt_db.ids
        print(len(self.ids))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]

        txt_inputs, txt_gts, txt_masks = self._get_txt_token(example['answer_ids'])

        if self.mode=='train':
            img_id = 'mscoco/train2014/'+'COCO_train2014_'+example['image_id'].zfill(12)
        else:
            img_id = 'mscoco/val2014/'+'COCO_val2014_'+example['image_id'].zfill(12)

        img_feat, img_pos_feat, _, image = self._get_img_feat(img_id)

        input_ids = np.array(example['input_ids'])
        input_ids = input_ids[:50]

        attn_masks = np.ones(img_feat.shape[0]+input_ids.shape[0], dtype=np.int64)

        ids = example['question_id']

        return (ids, input_ids, img_feat, img_pos_feat, attn_masks,
                txt_inputs, txt_gts, txt_masks, image)

def vqa_collate(inputs):
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
    (ids, input_ids, img_feats, img_pos_feats, attn_masks,
     txt_inputs, txt_gts, txt_masks, images) = map(list, unzip(inputs))

    txt_lens = [i.shape[0] for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64), 0)

    num_bbs = [f.shape[0] for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs, max_len=config.MAX_IMG_LEN)
    img_pos_feat = pad_tensors_pos(img_pos_feats, num_bbs, img_feat, max_len=config.MAX_IMG_LEN)
    images = stack_images(images)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=config.MAX_IMG_TEXT_LEN)
    sample_size = len(inputs[0])
    assert all(sample_size == len(i) for i in inputs)

    bs, max_tl = input_ids.shape
    out_size = attn_masks.shape[1]
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
    #gather_index = np.expand_dims(np.arange(0, out_size, dtype=np.int64), 0).repeat(batch_size, axis=0)


    # txt decoder
    txt_inputs = pad_sequence(txt_inputs, batch_first=True, padding_value=0)
    txt_gts = pad_sequence(txt_gts, batch_first=True, padding_value=0)
    txt_masks = pad_sequence(txt_masks, batch_first=True, padding_value=0)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'sample_size': sample_size,
             'images': images,
             'txt_inputs': txt_inputs,
             'txt_gts': txt_gts,
             'txt_masks': txt_masks,
             'ids': ids,}
    return batch