""" caption_data """

import numpy as np
from toolz.sandbox import unzip
import random
import os

from src.config import config
from src.data.utils import pad_sequence
from src.data.data_three import (TxtImgTwoDataset, TxtData, ImgData)
from src.data.randaugment import RandomAugment

import mindspore.dataset.vision as C_V
from mindspore.dataset.vision.utils import Inter
from PIL import Image

class CaptionDataset(TxtImgTwoDataset):
    """ CaptionDataset """

    def __init__(self, ids_path, txt_db, img_db):
        super().__init__(ids_path, txt_db, img_db)

        assert isinstance(txt_db, TxtData)

        self.cls_ = txt_db.cls_
        self.sep = txt_db.sep
        self.mask = txt_db.mask
        self.bos = 0
        self.eos = 0

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

        ids = example['img_id']

        # txt generate input and ground truth
        input_ids = example['input_ids']

        if input_ids is not None:
            txt_gts = np.array(input_ids[:config.MAX_TEXT_GTS_LEN] + [self.eos])
            attn_masks_text = np.ones(len(txt_gts),dtype=np.int64)
            input_ids = np.array([self.bos] + input_ids[:config.MAX_TEXT_GTS_LEN])

        else:
            input_ids = np.zeros(config.MAX_FULL_TEXT_LEN, dtype=np.int64)
            txt_gts = np.zeros(config.MAX_FULL_TEXT_LEN, dtype=np.int64)
            attn_masks_text = np.zeros(config.MAX_FULL_TEXT_LEN, dtype=np.int64)

        # img input
        image, patch_len = self._get_img_feat(example['img_id'])
        attn_masks_img = np.ones(patch_len, dtype=np.int64)

        attn_masks = np.ones(len(input_ids) + patch_len, dtype=np.int64)

        return (input_ids, txt_gts, attn_masks_text, image, attn_masks_img, attn_masks, ids)

def caption_collate(inputs):

    (input_ids, txt_gts, attn_masks_text, image, attn_masks_img, attn_masks, ids) = map(list, unzip(inputs))

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0, max_lens=config.MAX_IMG_TEXT_LEN)

    # txt decoder
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0, max_lens=config.MAX_FULL_TEXT_LEN)
    txt_gts = pad_sequence(txt_gts, batch_first=True, padding_value=0, max_lens=config.MAX_FULL_TEXT_LEN)
    attn_masks_text = pad_sequence(attn_masks_text, batch_first=True, padding_value=0, max_lens=config.MAX_FULL_TEXT_LEN)
    position_ids = np.expand_dims(np.arange(0, input_ids.shape[1], dtype=np.int64), 0)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'images': image,
             'txt_gts': txt_gts,
             'attn_masks': attn_masks,
             'attn_masks_text': attn_masks_text,
             'attn_masks_img':attn_masks_img,
             'ids': ids
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

class ImgDataTrain(ImgData):
    """ ImgData """

    def __init__(self, img_dir, opts):
        super().__init__(img_dir, opts)

    def get_image(self, id_):

        image_path = os.path.join(self.img_dir, id_)

        resize = self.image_size

        mean = [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
        std = [0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]

        trans = [
            C_V.RandomResizedCrop(resize, scale=(0.5, 1.0), interpolation=Inter.BILINEAR),
            C_V.RandomHorizontalFlip(),
            RandomAugment(2, 7, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            C_V.Normalize(mean=mean, std=std),
            C_V.HWC2CHW()
        ]

        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        for tran in trans:
            image = tran(image)

        return image