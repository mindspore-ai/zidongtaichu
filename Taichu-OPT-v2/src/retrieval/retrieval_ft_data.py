""" retrieval_data """

import numpy as np
from toolz.sandbox import unzip
import random

from src.config import config
from src.data.utils import pad_sequence
from src.data.data_three import (TxtImgTwoDataset, TxtData, ImgData)

class Re_Train_Dataset(TxtImgTwoDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """
    def __init__(self, ids_path, txt_db, img_db, use_mask_fix=False):
        super().__init__(ids_path, txt_db, img_db)

        assert isinstance(txt_db, TxtData)
        assert isinstance(img_db, ImgData)

        self.txt_db = txt_db
        self.img_db = img_db

    def _get_example(self, i):
        i = i % len(self.ids)
        anno = self.ids[i]
        example = {}

        img_id = anno.get('image',None)
        caption = anno.get('caption',None)

        if caption is not None and isinstance(caption[0], list):
            caption = random.choice(caption)

        example['input_ids'] = caption
        example['img_id'] = img_id

        return example

    def __getitem__(self, i):
        example = self._get_example(i)

        input_text = example['input_ids'][:config.MAX_TEXT_LEN]
        input_ids = self.txt_db.combine_inputs(input_text)
        attn_masks_text = np.ones(len(input_ids),dtype=np.int32)

         # img input
        image, patch_len = self._get_img_feat(example['img_id'])
        attn_masks_img = np.ones(patch_len, dtype=np.int32)

        attn_masks = np.ones(len(input_ids) + patch_len, dtype=np.int32)

        return (input_ids, attn_masks_text, image, attn_masks_img, attn_masks)

def retrieval_train_collate(inputs):

    (input_ids, attn_masks_text, image, attn_masks_img, attn_masks) = map(list, unzip(inputs))


    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=config.MAX_IMG_TEXT_LEN)

    # txt decoder
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0, max_lens=config.MAX_FULL_TEXT_LEN)
    attn_masks_text = pad_sequence(attn_masks_text, batch_first=True, padding_value=0, max_lens=config.MAX_FULL_TEXT_LEN)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64), 0)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'images': image,
             'attn_masks': attn_masks,
             'attn_masks_text': attn_masks_text,
             'attn_masks_img':attn_masks_img
            }

    return batch