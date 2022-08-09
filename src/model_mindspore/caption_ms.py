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
pretrain module
"""
import copy
import mindspore.nn as nn
from src.model_mindspore.parallel_transformer import ParallelConfig
from src.model_mindspore.model_config import UniterConfig
from src.model_mindspore.transformer_model import TransformerModel
from src.model_mindspore.model_ms import UniterThreeModel
from src.model_mindspore.loss import  TransformerTrainingLoss
from src.config.transformer_config import transformer_net_cfg as cfg

class UniterThreeForPretrainingForCapFinetune(nn.Cell):
    """ UNITER pretraining """

    def __init__(self, config, img_dim , audio_dim,
                 use_txt_out=False, use_video=False,  full_batch=True, use_moe=False, args=None):
         
        super(UniterThreeForPretrainingForCapFinetune, self).__init__()
        parallel_config = ParallelConfig()
        self.use_txt_out = use_txt_out
        self.use_video = use_video
        if self.use_video:
            img_dim = 1024
         
        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        config.use_vit = True 
        config.use_patch = True
        config.patch_size = 32
        config.train_image_size=448
        config.batch_size = args.train_batch_size
        self.uniter = UniterThreeModel(config, img_dim, audio_dim, use_video, parallel_config, use_moe)

        # Text Generator
        group_for_loss = 2
        fusion_group_backbone = parallel_config.fusion_group
        parallel_config.fusion_group = group_for_loss
        parallel_config.optimizer_shard = False
        fg_backbone = fusion_group_backbone // parallel_config.fusion_group
        self.txt_cfg = copy.deepcopy(cfg)
        self.txt_output = TransformerModel(self.txt_cfg, True, config.hidden_size, parallel_config, fg_backbone, False)
        self.td_crit = TransformerTrainingLoss(self.txt_cfg, parallel_config)

    def generate_text(self, sequence_output, att_masks, txt_gts, txt_masks):
        # generate text
        txt_out = self.txt_output(sequence_output, att_masks, txt_gts, txt_masks)
        loss = self.td_crit(txt_out, txt_gts, txt_masks)
        return loss

    def construct(self, input_ids, position_ids, img_feat, img_pos_feat, audio_feat, audio_pos_ids,
                  attention_mask, gather_index, txt_labels, txt_mask, txt_label_mask, img_mask_tgt,
                  img_mask_tgt_mask, img_masks, mrc_label_target, mrfr_feat_target,
                  audio_mask_tgt, audio_masks, mafr_feat_target, itm_targets, ma_neg_index, ma_neg_sample,
                  mr_neg_index, mr_neg_sample, txt_gts, txt_masks, img_token_gts, img_token_masks,images, images_mask,
                  taskId):
        """
        construct
        """
        sequence_output, _ , _= self.uniter(None, None,
                                         img_feat, img_pos_feat,
                                         attention_mask, gather_index,
                                         output_all_encoded_layers=False,
                                         txt_type_ids=None, img_type_ids=None,
                                         audio_feat=None, audio_pos_ids=None,
                                         audio_type_ids=None)

        td_loss = self.generate_text(sequence_output, attention_mask, txt_gts, txt_masks)

        return td_loss

class UniterThreeForPretrainingForCapFinetuneEval(nn.Cell):
    """ UNITER pretraining """

    def __init__(self, config, img_dim, audio_dim,
                 use_txt_out=False, use_video=False,  full_batch=True, use_moe=False, args=None,  beam_width=1):
         
        super(UniterThreeForPretrainingForCapFinetuneEval, self).__init__()
        parallel_config = ParallelConfig()
        self.use_txt_out = use_txt_out
        self.use_video = use_video
        if self.use_video:
            img_dim = 1024
         
        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        config.use_vit = True 
        config.use_patch = True
        config.patch_size = 32
        config.train_image_size=448
        config.batch_size = args.val_batch_size
        self.uniter = UniterThreeModel(config, img_dim, audio_dim, use_video, parallel_config, use_moe)

        # Text Generator
        group_for_loss = 2
        fusion_group_backbone = parallel_config.fusion_group
        parallel_config.fusion_group = group_for_loss
        parallel_config.optimizer_shard = False
        fg_backbone = fusion_group_backbone // parallel_config.fusion_group
        txt_cfg = copy.deepcopy(cfg)
        txt_cfg.batch_size = args.val_batch_size
        txt_cfg.beam_width = beam_width
        
        self.txt_output = TransformerModel(txt_cfg, False, config.hidden_size, parallel_config, fg_backbone, False)
        self.td_crit = TransformerTrainingLoss(txt_cfg, parallel_config)

    def generate_text_eval(self, sequence_output, att_masks):
        # generate text
        txt_out = self.txt_output(sequence_output, att_masks)
        return txt_out

    def construct(self, input_ids, position_ids, img_feat, img_pos_feat, audio_feat, audio_pos_ids,
                  attention_mask, gather_index, txt_labels, txt_mask, txt_label_mask, img_mask_tgt,
                  img_mask_tgt_mask, img_masks, mrc_label_target, mrfr_feat_target,
                  audio_mask_tgt, audio_masks, mafr_feat_target, itm_targets, ma_neg_index, ma_neg_sample,
                  mr_neg_index, mr_neg_sample, txt_gts, txt_masks, img_token_gts, img_token_masks,images, images_mask,
                  taskId):
        """
        construct
        """

        # if not self.full_batch:
        #     taskId = self.stride_slice_1(taskId, (0,), (1,), (1,))
        #     position_ids = self.stride_slice_2(position_ids, (0, 0), (1, 30), (1, 1))
        #     audio_pos_ids = self.stride_slice_2(audio_pos_ids, (0, 0), (1, 30), (1, 1))
        sequence_output, _, _ = self.uniter(None, None,
                                         img_feat, img_pos_feat,
                                         attention_mask, gather_index,
                                         output_all_encoded_layers=False,
                                         txt_type_ids=None, img_type_ids=None,
                                         audio_feat=None, audio_pos_ids=None,
                                         audio_type_ids=None)
         
        return self.generate_text_eval(sequence_output, attention_mask)

class UniterThreeForPretrainingForCapFinetuneInf(nn.Cell):
    """ UNITER pretraining """

    def __init__(self, config, img_dim, audio_dim,
                 use_txt_out=False, use_video=False, full_batch=True, use_moe=False, args=None, beam_width=1):
         
        super(UniterThreeForPretrainingForCapFinetuneInf, self).__init__()
        parallel_config = ParallelConfig()
        self.use_txt_out = use_txt_out
        self.use_video = use_video
        if self.use_video:
            img_dim = 1024
         
        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        config.use_vit = True 
        config.use_patch = True
        config.patch_size = 32
        config.train_image_size=448
        config.batch_size = 1
        self.uniter = UniterThreeModel(config, img_dim, audio_dim, use_video, parallel_config, use_moe)

        # Text Generator
        group_for_loss = 2
        fusion_group_backbone = parallel_config.fusion_group
        parallel_config.fusion_group = group_for_loss
        parallel_config.optimizer_shard = False
        fg_backbone = fusion_group_backbone // parallel_config.fusion_group
        txt_cfg = copy.deepcopy(cfg)
        txt_cfg.batch_size = 1
        txt_cfg.beam_width = beam_width
        txt_cfg.max_decode_length = 50

        self.txt_output = TransformerModel(txt_cfg, False, config.hidden_size, parallel_config, fg_backbone, False)
        self.td_crit = TransformerTrainingLoss(txt_cfg, parallel_config)

    def generate_text_eval(self, sequence_output, att_masks):
        # generate text
        return self.txt_output(sequence_output, att_masks)

    def construct(self, img_feat, img_pos_feat, attention_mask, gather_index):
        """
        construct
        """
        sequence_output, _, _ = self.uniter(None, None,
                                         img_feat, img_pos_feat,
                                         attention_mask, gather_index,
                                         output_all_encoded_layers=False,
                                         txt_type_ids=None, img_type_ids=None,
                                         audio_feat=None, audio_pos_ids=None,
                                         audio_type_ids=None)
         
        return self.generate_text_eval(sequence_output, attention_mask)