""" vqa_data """
import os
import numpy as np
from toolz.sandbox import unzip
from src.data.data_three import TxtImgTwoDataset, ImgData
from src.data.utils import pad_sequence
from src.config import config

import mindspore.dataset.vision as C_V
from mindspore.dataset.vision.utils import Inter
from PIL import Image

class VqaDataset(TxtImgTwoDataset):
    """ VqaDataset """

    def __init__(self, ids_path, txt_db, img_db):
        super().__init__(ids_path, txt_db, img_db)

    def _get_example(self, i):
        i = i % len(self.ids)
        anno = self.ids[i]
        example = {}

        image_id = anno.get('image_id',None)
        question = anno.get('Question',None)
        answer = anno.get('Answer',None)
        question_id = anno.get('question_id', None)

        example['question'] = question
        example['image_id'] = image_id
        example['answer'] = answer
        example['ids'] = question_id

        return example

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
        example = self._get_example(i)

        ids = example['ids']

        # txt generate input and ground truth
        input_text = example['question'][:config.MAX_TEXT_LEN]
        input_ids = self.txt_db.combine_inputs(input_text)
        attn_masks_text = np.ones(len(input_ids), dtype=np.int64)

        if example['answer'] is not None:
            txt_inputs, txt_gts, txt_gts_mask = self._get_txt_token(example['answer'])
            txt_gts_mask = np.array(txt_gts_mask, dtype=np.int64)

        else:
            txt_inputs = np.zeros(config.MAX_FULL_TEXT_LEN, dtype=np.int64)
            txt_gts = np.zeros(config.MAX_FULL_TEXT_LEN, dtype=np.int64)
            txt_gts_mask = np.zeros(config.MAX_FULL_TEXT_LEN, dtype=np.int64)

        # img input

        image, patch_len = self._get_img_feat(example['image_id'])
        attn_masks_img = np.ones(patch_len, dtype=np.int64)

        attn_masks = np.ones(len(input_ids) + patch_len, dtype=np.int64)

        return (input_ids, txt_gts, txt_gts_mask, attn_masks_text, image, attn_masks_img, attn_masks, ids)


def vqa_collate(inputs):

    (input_ids, txt_gts, txt_gts_mask, attn_masks_text, image, attn_masks_img, attn_masks, ids) = map(list, unzip(inputs))

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=config.MAX_IMG_TEXT_LEN)

    # txt decoder
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0, max_lens=config.MAX_FULL_TEXT_LEN)
    txt_gts = pad_sequence(txt_gts, batch_first=True, padding_value=0, max_lens=config.MAX_FULL_TEXT_LEN)
    txt_gts_mask = pad_sequence(txt_gts_mask, batch_first=True, padding_value=0, max_lens=config.MAX_FULL_TEXT_LEN)
    attn_masks_text = pad_sequence(attn_masks_text, batch_first=True, padding_value=0, max_lens=config.MAX_FULL_TEXT_LEN)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64), 0)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'images': image,
             'txt_gts': txt_gts,
             'txt_gts_mask': txt_gts_mask,
             'attn_masks': attn_masks,
             'attn_masks_text': attn_masks_text,
             'attn_masks_img':attn_masks_img,
             'ids':ids
            }

    return batch


class ImgDataEval(ImgData):
    """ ImgData """

    def __init__(self, img_dir, opts):
        self.img_dir = img_dir
        self.use_patch = opts.use_patch
        self.image_size = getattr(opts, "image_size", 224)
        self.patch_size = opts.patch_size

        print(f"image_size => {self.image_size} use_patch => {self.patch_size}")

    def get_image(self, id_):

        image_path = os.path.join(self.img_dir, id_)

        interpolation = "BILINEAR"

        mean = [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
        std = [0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]

        if hasattr(Inter, interpolation):
            interpolation = getattr(Inter, interpolation)
        else:
            interpolation = Inter.BILINEAR
            print('cannot find interpolation_type: {}, use {} instead'.format(interpolation, 'BILINEAR'))

        trans = [
            C_V.Resize(self.image_size, interpolation=Inter.BILINEAR),
            C_V.CenterCrop(self.image_size),
            C_V.Normalize(mean=mean, std=std),
            C_V.HWC2CHW()
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