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

UNITER pre-training
"""
import yaml

from src.data import (EasySampler, BatchSampler,
                  TxtData, ImgData, AudioData,
                  MlmThreeDataset, MrcThreeDataset, MrfrThreeDataset, ItmThreeDataset, MafrThreeDataset,
                  TdThreeDataset, TdOneDataset,
                  mlmThree_collate, mrcThree_collate, mrfrThree_collate, itmThree_collate, mafrThree_collate,
                  tdThree_collate, tdOne_collate,
                  IdThreeDataset, idThree_collate)
from src.data.retrieval_three import (TxttoImgEvalDataset, itmMatchingTxtImg_collate)
from src.data.caption_ft import CaptionDataset, caption_collate
from src.data.vqa import VQATxtData, VQATxtImgDataset, vqa_collate
from src.data.t2i_ft import T2IDetectFeatTxtTokTwoDataset, t2i_collate
from src.data.data_loader import DataLoader
from src.data.retrieval_ft import ItmFlickrRankDataset, itm_rank_collate
from src.data.ad_three import AdTextV3Dataset, adTextV3_collate
from src.tools.logger import LOGGER

def build_dataloader_ms(dataset, collate_fn, is_train, opts, device_num):
    if is_train:
        batch_size = opts.train_batch_size
    else:
        batch_size = opts.val_batch_size

    sampler = EasySampler(dataset, batch_size=batch_size, device_num=device_num)

    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, device_num=device_num)
    return loader

def build_dataloader_ft(dataset, collate_fn, is_train, opts, device_num):
    if is_train:
        batch_size = opts.train_batch_size
    else:
        batch_size = opts.val_batch_size
    sampler = BatchSampler(len(dataset), batch_size=batch_size, device_num=device_num)
    loader = DataLoader(dataset, batch_sampler=sampler,is_train=is_train,collate_fn=collate_fn, device_num=device_num,
                        drop_last=True)
    return loader

def build_dataloader_audio(dataset, collate_fn, device_num, batch_size=4):
    sampler = BatchSampler(len(dataset), batch_size=batch_size, device_num=device_num)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, device_num=device_num)
    return loader

# Masked Language Modeling
def build_mlmThree_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = MlmThreeDataset(ids_path, txt_db, img_db, audio_db,
                              use_mask_fix=opts.use_mask_fix)
    return dataset, mlmThree_collate


# Masked Region Classification
def build_mrcThree_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = MrcThreeDataset(ids_path, opts.mrm_prob, txt_db, img_db, audio_db, use_mask_fix=opts.use_mask_fix)
    return dataset, mrcThree_collate


# Masked Region Feature Regression (MRFR)
def build_mrfrThree_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = MrfrThreeDataset(opts.mrm_prob, ids_path, txt_db, img_db, audio_db,
                               use_mask_fix=opts.use_mask_fix)
    return dataset, mrfrThree_collate


# Masked Audio Feature Regression (MAFR)
def build_mafrThree_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = MafrThreeDataset(opts.mrm_prob, ids_path, txt_db, img_db, audio_db,
                               use_mask_fix=opts.use_mask_fix)
    return dataset, mafrThree_collate


# (ITM)
def build_itmThree_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = ItmThreeDataset(ids_path, txt_db, img_db, audio_db, opts.itm_neg_prob,
                              use_mask_fix=opts.use_mask_fix)
    return dataset, itmThree_collate


# Text Output
def build_tdThree_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = TdThreeDataset(ids_path, txt_db, img_db, audio_db,)
    return dataset, tdThree_collate


# Image Output
def build_idThree_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = IdThreeDataset(ids_path, txt_db, img_db, audio_db, opts.img_token_path, data_type=opts.data_type)
    return dataset, idThree_collate


# retrieval dataset
def build_t2i_dataset(ids_path, txt_db, img_db, audio_db, opts):
    dataset = TxttoImgEvalDataset(ids_path, txt_db, img_db, audio_db, use_mask_fix=opts.use_mask_fix)
    return dataset, itmMatchingTxtImg_collate

# Text One Decode Output DataLoader
def create_tdOne_dataloader(ids_path, img_dir, opts, device_num):
    dataset = TdOneDataset(ids_path, img_dir)
    loader = build_dataloader_ms(dataset, tdOne_collate, False, opts, device_num)
    dataloaders = {}
    dataloaders['tdOne'] = loader
    return dataloaders


def build_adText_dataset_v3(ids_path, txt_db, img_db, audio_db, opts):
    preprocess_config = yaml.load(open(opts.audio_preprocess_config, "r"), Loader=yaml.FullLoader)
    dataset = AdTextV3Dataset(ids_path, txt_db, opts.audio_mel_path, preprocess_config)
    return dataset, adTextV3_collate


def create_three_dataloaders(ids_path, datasets, is_train, opts, device_num, ids_two_path=None,
                             ids_textaudio_path=None):
    """ Create dataloaders """
    dataloaders = {}
    ## finetune Retrieval
    dset = datasets[0]
    if dset['tasks'][0].startswith('ftRet'):
        txt_db = TxtData(dset['db'][0])
        img_db = ImgData(dset['img'][0], opts)
        dataset = ItmFlickrRankDataset(ids_path, txt_db, img_db, neg_sample_size=1)
        loader = build_dataloader_ft(dataset, itm_rank_collate, is_train, opts, device_num)
        dataloaders["ftRet"] = loader
        return dataloaders, len(dataset)
    ## finetune caption
    if dset['tasks'][0].startswith('ftCap'):
        txt_db = TxtData(dset['db'][0])
        img_db = ImgData(dset['img'][0], opts)
        dataset = CaptionDataset(ids_path, txt_db, img_db)
        loader = build_dataloader_ft(dataset, caption_collate, is_train, opts, device_num)
        dataloaders["ftCap"] = loader
        return dataloaders, len(dataset)
    ## finetune vqa
    if dset['tasks'][0].startswith('vqa'):
        txt_db = VQATxtData(dset['db'][0], name=opts.name_txt, mode = opts.mode)
        img_db = ImgData(dset['img'][0], opts)
        dataset = VQATxtImgDataset(txt_db, img_db, opts.mode)
        loader = build_dataloader_ft(dataset, vqa_collate, is_train, opts, device_num)
        dataloaders["vqa"] = loader
        return dataloaders, len(dataset)
    ## t2i
    if dset['tasks'][0].startswith('ftT2I'):
        mode = 'train' if is_train is True else 'val'
        txt_db = TxtData(dset['db'][0])
        if is_train:
            batch_size = opts.train_batch_size
        else:
            batch_size = opts.val_batch_size
        dataset = T2IDetectFeatTxtTokTwoDataset(ids_path=ids_path, img_token_path=opts.img_token_path,
                                                max_txt_len=opts.max_txt_len, txt_db=txt_db,
                                                mode=mode, dataname=dset['name'], batch_size=batch_size)
        loader = build_dataloader_ft(dataset, t2i_collate, is_train, opts, device_num)
        dataloaders["ftT2I"] = loader
        return dataloaders, len(dataset)

    ## audio decoder
    if dset['tasks'][0].startswith('adTextEval'):
        txt_db = TxtData(dset['db'][0])
        img_db = ImgData(dset['img'][0], opts)
        audio_db = AudioData(dset['audio'][0])
        dataset, collate_fn = build_adText_dataset_v3(ids_path, txt_db, img_db, audio_db, opts)

        batch_size = opts.train_batch_size if is_train else opts.val_batch_size

        loader = build_dataloader_audio(dataset, collate_fn, device_num, batch_size=batch_size)

        dataloaders["adTextEval"] = loader
        return dataloaders, len(dataset)

    ## pretrain tasks
    for dset in datasets:
        if dset['tasks']:  # if the list sequence is empty, then it is equal to False
            txt_db = TxtData(dset['db'][0])
            img_db = ImgData(dset['img'][0], opts)
            audio_db = AudioData(dset['audio'][0])

        for i, t in enumerate(dset['tasks']):
            task = f'{t}_{dset["name"]}'
            if task.startswith('mlmThree'):
                dataset, collate_fn = build_mlmThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('mrcThree'):
                dataset, collate_fn = build_mrcThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('mrfrThree'):
                dataset, collate_fn = build_mrfrThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('mrctThree'):
                dataset, collate_fn = build_mrfrThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('itmThree'):
                dataset, collate_fn = build_itmThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('mafrThree'):
                dataset, collate_fn = build_mafrThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('macThree'):
                dataset, collate_fn = build_mafrThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('tdThree'):
                dataset, collate_fn = build_tdThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('idThree'):
                dataset, collate_fn = build_idThree_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('ret_t2i'):
                dataset, collate_fn = build_t2i_dataset(ids_path, txt_db, img_db, audio_db, opts)
            elif task.startswith('adText'):
                dataset, collate_fn = build_adText_dataset_v3(ids_path, txt_db, img_db, audio_db, opts)
            else:
                raise ValueError('Undefined task %s'% (task))
            LOGGER.info("Create Dataset %s Success", (task))
            if task.startswith('ret'):
                loader = build_dataloader_ft(dataset, collate_fn, is_train, opts, device_num)
            else:
                loader = build_dataloader_ms(dataset, collate_fn, is_train, opts, device_num)

            if is_train:
                ratio = dset['mix_ratio'][i]
                dataloaders[task] = (loader, ratio)
            else:
                dataloaders[task] = loader
    return dataloaders, len(dataset)
