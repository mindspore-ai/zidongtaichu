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
vqa model
"""
import copy

import mindspore.nn as nn
from src.model_mindspore.parallel_transformer import ParallelConfig
from src.model_mindspore.model_config import UniterConfig
from src.model_mindspore.transformer_model import TransformerModel
from src.model_mindspore.model_ms import UniterThreeModel
from src.model_mindspore.loss import TransformerTrainingLoss
from src.config.transformer_config import transformer_net_cfg as cfg

class UniterThreeForPretrainingForVQAFinetune(nn.Cell):
    """ UNITER VQAGEN FINETUNE """

    def __init__(self, config, img_dim, img_label_dim, audio_dim, audio_label_dim,
                 use_txt_out=False, use_video=False, full_batch=True, use_moe=False, is_parallel=True, args=None):
        super(UniterThreeForPretrainingForVQAFinetune, self).__init__()
        parallel_config = ParallelConfig()
        
        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        config.batch_size = args.train_batch_size
        self.uniter = UniterThreeModel(config, img_dim, audio_dim, use_video, parallel_config, use_moe, is_parallel)
        
        group_for_loss = 2
        fusion_group_backbone = parallel_config.fusion_group
        parallel_config.fusion_group = group_for_loss
        parallel_config.optimizer_shard = False
        fg_backbone = fusion_group_backbone // parallel_config.fusion_group
        txt_cfg = copy.deepcopy(cfg)
        self.txt_output = TransformerModel(txt_cfg, True, config.hidden_size, parallel_config, fg_backbone, False)
        self.td_crit = TransformerTrainingLoss(txt_cfg, parallel_config)
        
    def generate_text(self, sequence_output, att_masks, txt_gts, txt_masks):
        """Generate text"""
        txt_out = self.txt_output(sequence_output, att_masks, txt_gts, txt_masks)
        loss = self.td_crit(txt_out, txt_gts, txt_masks)
        return loss
    
    def construct(self, input_ids, position_ids, img_feat, img_pos_feat, audio_feat, audio_pos_ids,
                  attention_mask, gather_index, txt_labels, txt_mask, txt_label_mask, img_mask_tgt,
                  img_mask_tgt_mask, img_masks, mrc_label_target, mrfr_feat_target,
                  audio_mask_tgt, audio_masks, mafr_feat_target, itm_targets, ma_neg_index, ma_neg_sample,
                  mr_neg_index, mr_neg_sample, txt_gts, txt_masks, img_token_gts, img_token_masks, images, images_mask,
                  taskId):
        """
        construct
        """
        sequence_output, _, _ = self.uniter(input_ids, position_ids,
                                          img_feat, img_pos_feat,
                                          attention_mask, gather_index, img_masks=img_masks,
                                          output_all_encoded_layers=False,
                                          txt_type_ids=None, img_type_ids=None,
                                          audio_feat=None, audio_pos_ids=None,
                                          audio_type_ids=None, audio_masks=None,
                                          images=None)

        td_loss = self.generate_text(sequence_output, attention_mask, txt_gts, txt_masks)

        return td_loss

class UniterThreeForPretrainingForVQAFinetuneEval(nn.Cell):
    """ UNITER pretraining """

    def __init__(self, config, img_dim, img_label_dim, audio_dim, audio_label_dim,
                 use_txt_out=False, use_video=False, full_batch=True, use_moe=False, is_parallel=True, args=None):
        super(UniterThreeForPretrainingForVQAFinetuneEval, self).__init__()
        parallel_config = ParallelConfig()
        
        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        config.batch_size = args.val_batch_size
        self.uniter = UniterThreeModel(config, img_dim, audio_dim, use_video, parallel_config, use_moe, is_parallel)

        #Text Generator
        group_for_loss = 2
        fusion_group_backbone = parallel_config.fusion_group
        parallel_config.fusion_group = group_for_loss
        parallel_config.optimizer_shard = False
        fg_backbone = fusion_group_backbone // parallel_config.fusion_group
        txt_cfg = copy.deepcopy(cfg)
        txt_cfg.batch_size = args.val_batch_size
        txt_cfg.beam_width = args.beam_width
        self.txt_output = TransformerModel(txt_cfg, False, config.hidden_size, parallel_config, fg_backbone, False)

    def generate_text_eval(self, sequence_output, att_masks):
        # generate text
        txt_out = self.txt_output(sequence_output, att_masks)
        return txt_out

    def construct(self, input_ids, position_ids, img_feat, img_pos_feat, audio_feat, audio_pos_ids,
                  attention_mask, gather_index, txt_labels, txt_mask, txt_label_mask, img_mask_tgt,
                  img_mask_tgt_mask, img_masks, mrc_label_target, mrfr_feat_target,
                  audio_mask_tgt, audio_masks, mafr_feat_target, itm_targets, ma_neg_index, ma_neg_sample,
                  mr_neg_index, mr_neg_sample, txt_gts, txt_masks, img_token_gts, img_token_masks, images, images_mask,
                  taskId):
        """
        construct
        """
        sequence_output, _, _ = self.uniter(input_ids, position_ids,
                                          img_feat, img_pos_feat,
                                          attention_mask, gather_index, img_masks=img_masks,
                                          output_all_encoded_layers=False,
                                          txt_type_ids=None, img_type_ids=None,
                                          audio_feat=None, audio_pos_ids=None,
                                          audio_type_ids=None, audio_masks=None,
                                          images=None)

        ans = self.generate_text_eval(sequence_output, attention_mask)
        return ans

class UniterThreeForPretrainingForVQAFinetuneInf(nn.Cell):
    """ UNITER pretraining """

    def __init__(self, config, img_dim, img_label_dim, audio_dim, audio_label_dim,
                 use_txt_out=False, use_video=False, full_batch=True, use_moe=False, is_parallel=True, args=None, beam_width=1):
        super(UniterThreeForPretrainingForVQAFinetuneInf, self).__init__()
        parallel_config = ParallelConfig()
        
        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        config.batch_size = 1
        self.uniter = UniterThreeModel(config, img_dim, audio_dim, use_video, parallel_config, use_moe, is_parallel)

        #Text Generator
        group_for_loss = 2
        fusion_group_backbone = parallel_config.fusion_group
        parallel_config.fusion_group = group_for_loss
        parallel_config.optimizer_shard = False
        fg_backbone = fusion_group_backbone // parallel_config.fusion_group
        txt_cfg = copy.deepcopy(cfg)
        txt_cfg.batch_size = 1
        txt_cfg.beam_width = beam_width
        self.txt_output = TransformerModel(txt_cfg, False, config.hidden_size, parallel_config, fg_backbone, False)

    def generate_text_eval(self, sequence_output, att_masks):
        # generate text
        txt_out = self.txt_output(sequence_output, att_masks)
        return txt_out

    def construct(self, input_ids, position_ids, img_feat, img_pos_feat,
                  attention_mask, gather_index):
        """
        construct
        """
        sequence_output, _, _ = self.uniter(input_ids, position_ids,
                                          img_feat, img_pos_feat,
                                          attention_mask, gather_index, img_masks=None,
                                          output_all_encoded_layers=False,
                                          txt_type_ids=None, img_type_ids=None,
                                          audio_feat=None, audio_pos_ids=None,
                                          audio_type_ids=None, audio_masks=None,
                                          images=None)

        ans = self.generate_text_eval(sequence_output, attention_mask)
        return ans
