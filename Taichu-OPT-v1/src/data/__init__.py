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
from .sampler import EasySampler, BatchSampler
from .loader import MetaLoader, MetaLoaderAudio, task2id
from .generator import data_column, data_column_audio, get_batch_data, get_batch_data_audio

from .data_three import (TxtData, ImgData, AudioData)
from .mlm_three import MlmThreeDataset, mlmThree_collate
from .mrm_three import (MrfrThreeDataset, MrcThreeDataset,
                        mrfrThree_collate, mrcThree_collate)
from .itm_three import ItmThreeDataset, itmThree_collate
from .mam_three import MafrThreeDataset, mafrThree_collate
from .td_three import TdThreeDataset, tdThree_collate, TdOneDataset, tdOne_collate
from .id_three import IdThreeDataset, idThree_collate
from .retrieval_three import (itmMatchingTxtImg_collate,
                              TxttoImgEvalDataset)
from .caption_ft import CaptionDataset, caption_collate
from .ad_three import AdTextV3Dataset, adTextV3_collate

from .dataset import create_dataset, create_audio_dataset
from .vqa import VQATxtData, VQATxtImgDataset, vqa_collate
