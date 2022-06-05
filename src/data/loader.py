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
"""loader"""
import random
import time
from collections import defaultdict
from multiprocessing import Process
import numpy as np
from src.data.data_loader import DataLoader
from src.config import config

data_column = [
    'input_ids',
    'position_ids',
    'img_feat',
    'img_pos_feat',
    'audio_feat',
    'audio_pos_ids',
    'attention_mask',
    'gather_index',
    'txt_labels',
    'txt_mask',
    'txt_label_mask',
    'img_mask_tgt',
    'img_mask_tgt_mask',
    'img_masks',
    'mrc_label_target',
    'mrfr_feat_target',
    'audio_mask_tgt_mask',
    'audio_masks',
    'mafr_feat_target',
    'itm_target',
    'ma_neg_index',
    'ma_neg_sample',
    'mr_neg_index',
    'mr_neg_sample',
    'txt_gts',
    'txt_masks',
    'img_token_gts',
    'img_token_masks',
    'images',
    'images_mask',
    'taskId'
]

# loss = self.concat((mlm_loss.view(1,), mafr_loss.view(1,), mrfr_loss.view(1,),
#                    mac_loss.view(1,), itm_loss.view(1,), td_loss.view(1,), id_loss.view(1,)))

task2id = {
    'mlmThree': 0,
    'mafrThree': 1,
    'mrfrThree': 2,
    'macThree': 3,
    "itmThree": 4,
    "tdThree": 5,
    "idThree": 6,
    "adThree": 7,
    "ret": 10,
    "ftRet": 11,
    "ftCap": 12,
    "vqa": 13,
    "ftT2I": 14,
}

class MetaLoader():
    """ wraps multiple data loaders """

    def __init__(self, loaders, datalen, accum_steps=1, task_num=9):
        assert isinstance(loaders, dict)
        self.task_num = task_num
        self.name2loader = {}
        self.name2iter = {}
        self.name2iter_copy = {}
        self.sampling_pools = []
        self.loaders = loaders
        self.datalen = datalen
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, DataLoader):
                r = 1
            else:
                raise ValueError()
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.sampling_pools.extend([n] * r)

        self.task = self.sampling_pools[0]
        self.task_label = [0] * self.task_num
        self.step = 0
        self.accum_steps = accum_steps
        self.step_cnt = 0
        self.task_index_list = np.random.permutation(self.task_num)
        random.seed(1)
        self.all_ids = []

    def init_iter(self, task_name):
        self.name2iter[task_name] = iter(self.name2loader[task_name])

    def return_ids(self):
        return self.all_ids

    def get_batch_params(self, batch):
        """ get_batch_params """

        batch = defaultdict(lambda: None, batch)

        input_ids = batch.get('input_ids', None)
        position_ids = batch.get('position_ids', None)

        img_feat = batch['img_feat']  # self.bs, 10,d 2048
        img_pos_feat = batch['img_pos_feat']  # self.bs, 10, 7

        audio_feat = batch['audio_feat']  # self.bs, 10, 512
        audio_pos_ids = batch['audio_pos_ids']  # 1, 10

        # attention_mask: 32 * 191
        attention_mask = batch['attn_masks']
        # gather_index 32 * 191
        gather_index = batch['gather_index']

        txt_labels = batch['txt_labels']
        txt_mask = batch['txt_mask']
        txt_label_mask = batch['txt_label_mask']

        img_mask_tgt = batch['img_mask_tgt']  # self.bs, 72
        img_mask_tgt_mask = batch['img_mask_tgt_mask']  # self.bs*2, 2
        img_masks = batch['img_masks']  # self.bs, 10
        mrc_label_target = batch['label_targets']  # self.bs*2, 1

        audio_mask_tgt_mask = batch['audio_mask_tgt_mask']
        audio_masks = batch['audio_masks']

        mrfr_feat_target = batch.get('mrfr_feat_target', None)
        mafr_feat_target = batch.get('mafr_feat_target', None)

        itm_target = batch.get('targets', None)

        ma_neg_index = batch.get('ma_neg_index', None)
        ma_neg_sample = batch.get('ma_neg_sample', None)
        mr_neg_index = batch.get('mr_neg_index', None)
        mr_neg_sample = batch.get('mr_neg_sample', None)

        txt_gts = batch.get('txt_gts', None)
        txt_masks = batch.get('txt_masks', None)

        img_token_gts = batch.get('img_token_gts', None)
        img_token_masks = batch.get('img_token_masks', None)

        images = batch.get('images', None)
        images_mask = batch.get('images_mask', None)

        return (input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
                audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
                txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
                mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
                ma_neg_index, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
                txt_masks, img_token_gts, img_token_masks, images, images_mask)

    def get_batch_check(self, batch, input_ids, position_ids, audio_feat,
                        audio_pos_ids, attention_mask, txt_labels, txt_mask,
                        txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
                        mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
                        ma_neg_index, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
                        txt_masks, img_token_gts, img_token_masks, images, images_mask):
        """ get_batch_check """

        mask_size = config.MASK_SIZE
        n_negatives = config.N_NEGATIVES
        ids = batch.get('ids', None)
        if ids is not None:
            self.all_ids = self.all_ids + ids
        self.bs = attention_mask.shape[0]  # add by zjzhao
        #         print("self.bs=========================", self.bs)
        # text
        if input_ids is None:
            input_ids = np.zeros((self.bs, config.MAX_FULL_TEXT_LEN)).astype(np.int32)
        if position_ids is None:
            position_ids = np.zeros((1, config.MAX_FULL_TEXT_LEN)).astype(np.int32)
        if txt_labels is None:
            txt_labels = np.zeros((self.bs, config.MAX_FULL_TEXT_LEN)).astype(np.int32)
        if txt_mask is None:
            txt_mask = np.zeros((self.bs * mask_size, 2)).astype(np.int32)
        if txt_label_mask is None:
            txt_label_mask = np.zeros(self.bs * mask_size).astype(np.int32)

        # image
        if img_mask_tgt is None:
            img_mask_tgt = np.zeros((self.bs, config.MAX_FULL_LEN)).astype(np.bool_)
        if img_mask_tgt_mask is None:
            img_mask_tgt_mask = np.zeros((self.bs * mask_size, 2)).astype(np.int32)
        if img_masks is None:
            img_masks = np.zeros((self.bs, config.MAX_IMG_LEN)).astype(np.bool_)
        if mrc_label_target is None:
            mrc_label_target = np.zeros((self.bs * mask_size, 1)).astype(np.float32)

        # audio
        if audio_feat is None:
            audio_feat = np.zeros((self.bs, config.MAX_AUDIO_LEN, config.AUDIO_DIM)).astype(
                np.float32)  # 用attention_mask.shape[0]替换了self.bs
        if audio_pos_ids is None:
            audio_pos_ids = np.zeros((1, config.MAX_AUDIO_LEN)).astype(np.int32)

        if mrfr_feat_target is None:
            mrfr_feat_target = np.zeros((self.bs * mask_size, config.IMG_DIM)).astype(np.float32)

        if audio_mask_tgt_mask is None:
            audio_mask_tgt_mask = np.zeros((self.bs * mask_size, 2)).astype(np.int32)
        if audio_masks is None:
            audio_masks = np.zeros((self.bs, config.MAX_AUDIO_LEN)).astype(np.bool_)

        if mafr_feat_target is None:
            mafr_feat_target = np.zeros((self.bs * mask_size, 1024)).astype(np.float32)

        if itm_target is None:
            itm_target = np.zeros((self.bs,)).astype(np.int32)
        if ma_neg_index is None:
            ma_neg_index = np.zeros((self.bs * mask_size, 1)).astype(np.int32)
        if ma_neg_sample is None:
            ma_neg_sample = np.zeros((self.bs * mask_size, n_negatives, config.AUDIO_DIM)).astype(np.float32)

        if mr_neg_index is None:
            mr_neg_index = np.zeros((self.bs * mask_size, 1)).astype(np.int32)
        if mr_neg_sample is None:
            mr_neg_sample = np.zeros((self.bs * mask_size, n_negatives, config.IMG_DIM)).astype(np.float32)
        if txt_gts is None:
            txt_gts = np.zeros((self.bs, config.MAX_FULL_TEXT_LEN)).astype(np.int32)
        if txt_masks is None:
            txt_masks = np.ones((self.bs, config.MAX_FULL_TEXT_LEN)).astype(np.float32)

        if img_token_gts is None:
            img_token_gts = np.zeros((self.bs, config.IMG_TOKEN_LEN)).astype(np.int32)
        if img_token_masks is None:
            img_token_masks = np.ones((self.bs, config.IMG_TOKEN_LEN)).astype(np.float32)
        if images is None:
            images = np.ones((self.bs, 3, config.IMG_PATCH_SIZE, config.IMG_PATCH_SIZE)).astype(np.float32)
        if images_mask is None:
            images_mask = np.zeros((self.bs * mask_size, 2)).astype(np.int32)

        return (input_ids, position_ids, audio_feat,
                audio_pos_ids, attention_mask, txt_labels, txt_mask,
                txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
                mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
                ma_neg_index, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
                txt_masks, img_token_gts, img_token_masks, images, images_mask)

    def get_batch(self, batch, task):
        """ get_batch """

        (input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
         audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
         txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
         mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
         ma_neg_index, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
         txt_masks, img_token_gts, img_token_masks, images, images_mask) = self.get_batch_params(batch)

        (input_ids, position_ids, audio_feat,
         audio_pos_ids, attention_mask, txt_labels, txt_mask,
         txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
         mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
         ma_neg_index, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
         txt_masks, img_token_gts, img_token_masks, images, images_mask) = self.get_batch_check(batch, input_ids,
                                                                                                position_ids,
                                                                                                audio_feat,
                                                                                                audio_pos_ids,
                                                                                                attention_mask,
                                                                                                txt_labels,
                                                                                                txt_mask,
                                                                                                txt_label_mask,
                                                                                                img_mask_tgt,
                                                                                                img_mask_tgt_mask,
                                                                                                img_masks,
                                                                                                mrc_label_target,
                                                                                                mrfr_feat_target,
                                                                                                audio_mask_tgt_mask,
                                                                                                audio_masks,
                                                                                                mafr_feat_target,
                                                                                                itm_target,
                                                                                                ma_neg_index,
                                                                                                ma_neg_sample,
                                                                                                mr_neg_index,
                                                                                                mr_neg_sample, txt_gts,
                                                                                                txt_masks,
                                                                                                img_token_gts,
                                                                                                img_token_masks,
                                                                                                images, images_mask)

        # if self.print_time:
        #     print("txt: {} img:{} audio:{}".format(input_ids.shape, img_feat.shape, audio_feat.shape))
        taskId = np.array([task2id[task]]).astype(np.int32)
        txt_masks = txt_masks.astype(np.float32)

        # print("task:{} input_ids:{}, position_ids:{}, img_feat:{}, img_pos_feat:{}, audio_feat:{}, \
        #           audio_pos_ids:{}, attention_mask:{}, gather_index:{}, txt_labels:{}, txt_mask:{}, \
        #           txt_label_mask:{}, img_mask_tgt:{}, img_mask_tgt_mask:{}, img_masks:{}, mrc_label_target:{}, \
        #           mrfr_feat_target:{}, audio_mask_tgt_mask:{}, audio_masks:{}, mafr_feat_target:{}, itm_target:{}, \
        #           txt_label_mask:{}, ma_neg_sample:{}, mr_neg_index:{}, mr_neg_sample:{}, txt_gts:{}, \
        #           txt_masks:{}, img_token_gts:{}, img_token_masks:{}, images:{}, images_mask:{}".format(
        #      task, input_ids.shape, position_ids.shape, img_feat.shape, img_pos_feat.shape, audio_feat.shape,
        #           audio_pos_ids.shape, attention_mask.shape, gather_index.shape, txt_labels.shape, txt_mask.shape,
        #           txt_label_mask.shape, img_mask_tgt.shape, img_mask_tgt_mask.shape, img_masks.shape, mrc_label_target.shape,
        #           mrfr_feat_target.shape, audio_mask_tgt_mask.shape, audio_masks.shape, mafr_feat_target.shape, itm_target.shape,
        #           txt_label_mask.shape, ma_neg_sample.shape, mr_neg_index.shape, mr_neg_sample.shape, txt_gts.shape,
        #           txt_masks.shape, img_token_gts.shape, img_token_masks.shape, images.shape, images_mask.shape))
        # print("task:{} input_ids:{}, position_ids:{}, img_feat:{}, img_pos_feat:{}, audio_feat:{}, \
        #           audio_pos_ids:{}, attention_mask:{}, gather_index:{}, txt_labels:{}, txt_mask:{}, \
        #           txt_label_mask:{}, img_mask_tgt:{}, img_mask_tgt_mask:{}, img_masks:{}, mrc_label_target:{}, \
        #           mrfr_feat_target:{}, audio_mask_tgt_mask:{}, audio_masks:{}, mafr_feat_target:{}, itm_target:{}, \
        #           txt_label_mask:{}, ma_neg_sample:{}, mr_neg_index:{}, mr_neg_sample:{}, txt_gts:{}, \
        #           txt_masks:{}, img_token_gts:{}, img_token_masks:{}, images:{}, images_mask:{}".format(
        #             task, input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
        #             audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
        #             txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
        #             mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
        #             txt_label_mask, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
        #             txt_masks, img_token_gts, img_token_masks, images, images_mask))


        output = (input_ids, position_ids, img_feat, img_pos_feat, audio_feat,
                  audio_pos_ids, attention_mask, gather_index, txt_labels, txt_mask,
                  txt_label_mask, img_mask_tgt, img_mask_tgt_mask, img_masks, mrc_label_target,
                  mrfr_feat_target, audio_mask_tgt_mask, audio_masks, mafr_feat_target, itm_target,
                  txt_label_mask, ma_neg_sample, mr_neg_index, mr_neg_sample, txt_gts,
                  txt_masks, img_token_gts, img_token_masks, images, images_mask, taskId)

        return output

    def __getitem__(self, index):
        start_time = time.time()
        if self.step_cnt == self.task_num:
            self.task_index_list = np.random.permutation(self.task_num)
            self.step_cnt = 0
        task_index = self.task_index_list[self.step_cnt]
        local_task = self.sampling_pools[task_index]

        iter_ = self.name2iter[local_task]

        name = local_task
        try:
            batch = next(iter_)
        except StopIteration:
            print("============EPOCH END=============", flush=True)
            self.init_iter(local_task)
            print("cost init iter time :", time.time() - start_time, flush=True)
            iter_ = self.name2iter[local_task]
            batch = next(iter_)


        task = name.split('_')[0]
        # Prepare Data
        for key, val in batch.items():
            if isinstance(val, np.ndarray):
                if val.dtype == np.int64:
                    batch[key] = val.astype(np.int32)

        output = self.get_batch(batch, task)

        # if self.print_time:
        #     print("============index: {}, taskname: {}, costtime: {}".format(index, task, time.time() - start_time))
        self.step_cnt += 1
        return output

    def __len__(self):
        # return 180 216 11961(256)  47853 83745(128 300w)   1314(128 100000) 5672*9 3545*9
        # return 5672*9
        return self.datalen


class MetaLoaderAudio:
    """ wraps multiple data loaders """

    def __init__(self, loaders, datalen, accum_steps=1, task_num=9):
        assert isinstance(loaders, dict)
        self.task_num = task_num
        self.name2loader = {}
        self.name2iter = {}
        self.name2iter2 = {}
        self.sampling_pools = []
        self.loaders = loaders
        self.datalen = datalen
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, DataLoader):
                r = 1
            else:
                raise ValueError()
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.name2iter2[n] = iter(l)
            self.sampling_pools.extend([n] * r)
        self.task = self.sampling_pools[0]
        self.task_label = [0] * self.task_num
        # self.task_label = [0] * 10
        self.step = 0
        self.accum_steps = accum_steps
        self.step_cnt = 0
        self.flag = "iter1"
        self.iter1_init_cnt = 0
        self.iter2_init_cnt = 0
        # self.task_index_list = np.random.permutation(10)
        self.task_index_list = np.random.permutation(self.task_num)
        random.seed(1)
        self.all_ids = []

    def init_iter(self, task_name):
        self.name2iter[task_name] = iter(self.name2loader[task_name])

    def return_ids(self):
        return self.all_ids

    def __getitem__(self, index):

        start_time = time.time()

        if self.step_cnt == self.task_num:
            self.task_index_list = np.random.permutation(self.task_num)
            self.step_cnt = 0
        self.step_cnt += 1
        task_index = self.task_index_list[self.step_cnt - 1]
        local_task = self.sampling_pools[task_index]

        iter_ = self.name2iter[local_task]

        name = local_task
        try:
            batch = next(iter_)
        except StopIteration:
            print("============EPOCH END=============", flush=True)
            self.init_iter(local_task)
            print("cost init iter time :", time.time() - start_time, flush=True)
            iter_ = self.name2iter[local_task]
            batch = next(iter_)

        task = name.split('_')[0]

        # Prepare Data
        for key, val in batch.items():
            if isinstance(val, np.ndarray):
                if val.dtype == np.int64:
                    batch[key] = val.astype(np.int32)

        batch = defaultdict(lambda: None, batch)

        input_ids = batch.get('input_ids', None)
        position_ids = batch.get('position_ids', None)

        # attention_mask: 32 * 191
        attention_mask = batch['attn_masks']

        mel_targets = batch['audio_mel_targets']
        duration_targets = batch['audio_duration_targets']
        speakers = batch['audio_speakers']
        texts = batch['audio_texts']
        src_lens = batch['audio_text_lens']
        mel_lens = batch['audio_mel_lens']
        audio_max_text_len = batch['audio_max_text_len']
        audio_max_mel_len = batch['audio_max_mel_len']
        pitch_targets = batch['audio_pitch_targets']
        energy_targets = batch['audio_energy_targets']

        ids = batch.get('ids', None)
        if ids is not None:
            self.all_ids = self.all_ids + ids

        output = (input_ids, position_ids, attention_mask,
                  mel_targets, duration_targets, speakers, texts, src_lens, mel_lens,
                  audio_max_text_len, audio_max_mel_len, pitch_targets, energy_targets)

        # print("============index: {}, taskname: {}, costtime: {}".format(index, task, time.time() - start_time))
        return output

    def __len__(self):
        # return 256*64
        return self.datalen
