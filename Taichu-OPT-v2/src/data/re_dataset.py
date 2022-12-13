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
"""retrieval three"""
import os
import json
from toolz.sandbox import unzip
import numpy as np
from src.config import config
from src.data.utils import pad_sequence
from src.data.data_three import stack_images, get_image_from_path

class Re_Eval_Dataset():
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """
    def __init__(self, ann_file, image_root):
        self.ann = json.load(open(ann_file,'r'))
        self.image_root = image_root

        self.cls_ = 101
        self.sep = 102
        self.mask = 103
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            caption = ann['caption']
            if not isinstance(caption[0], list):
                caption = [caption]
            for i, cap in enumerate(caption):
                self.text.append(cap)
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

        self.num = len(self.text)//len(self.image)
        print(f"IMAGE NUM: {len(self.image)} TEXT NUM: {len(self.text)}")


class Re_Eval_Img_Dataset(Re_Eval_Dataset):
    def __init__(self, ann_file, image_root):
        super().__init__(ann_file, image_root)
        print(f"IMG_SIZE: {config.IMG_SIZE} IMG_PATCH_SIZE: {config.IMG_PATCH_SIZE}")

    def __len__(self):
        return len(self.image)

    def __getitem__(self, i):
        image_path = os.path.join(self.image_root, self.ann[i]['image'])     
        image, patch_len = get_image_from_path(image_path, image_size=config.IMG_SIZE, patch_size=config.IMG_PATCH_SIZE)
        input_ids = [self.cls_] + self.text[i*self.num][:config.MAX_TEXT_LEN] + [self.sep]
        input_ids = np.array(input_ids)
        attn_masks_text = np.ones(len(input_ids), dtype=np.int64)
        attn_masks = np.ones(patch_len, dtype=np.int64)
        return (input_ids, attn_masks, image, attn_masks_text)

class Re_Eval_Txt_Dataset(Re_Eval_Dataset):
    def __init__(self, ann_file, image_root):
        super().__init__(ann_file, image_root)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        image_path = os.path.join(self.image_root, self.ann[i//self.num]['image'])
        image, patch_len = get_image_from_path(image_path, image_size=config.IMG_SIZE, patch_size=config.IMG_PATCH_SIZE)
        input_ids = [self.cls_] + self.text[i][:config.MAX_TEXT_LEN] + [self.sep]
        input_ids = np.array(input_ids)
        attn_masks = np.ones(len(input_ids) + patch_len, dtype=np.int64)
        attn_masks_text = np.ones(len(input_ids), dtype=np.int64)

        return (input_ids, attn_masks, image, attn_masks_text)

def itmTwo_collate(inputs):
    """ itmThree_collate """

    (input_ids, attn_masks, images, attn_masks_text) = map(list, unzip(inputs))

    # text batches
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0, max_lens=config.MAX_FULL_TEXT_LEN)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64), 0)

    # image batches
    images = stack_images(images)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=config.MAX_IMG_TEXT_LEN)
    attn_masks_text = pad_sequence(attn_masks_text, batch_first=True, padding_value=0, max_lens=config.MAX_FULL_TEXT_LEN)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'attn_masks': attn_masks,
             'images': images,
             'attn_masks_text': attn_masks_text}
    return batch