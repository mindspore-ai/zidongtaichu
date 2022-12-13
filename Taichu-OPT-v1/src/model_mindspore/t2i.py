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
import copy
import mindspore.nn as nn
from src.model_mindspore.parallel_transformer import ParallelConfig
from src.model_mindspore.model_config import UniterConfig
from src.model_mindspore.transformer_model import TransformerModel
from src.model_mindspore.model_ms import UniterThreeModel
from src.model_mindspore.loss import  TransformerTrainingLoss
from src.config.transformer_config import transformer_net_cfg as cfg
from src.mae_mindspore.src.vit import Vit


class UniterThreeForPretrainingForT2Ifinetune(nn.Cell):
    """ UNITER pretraining """

    def __init__(self, config, img_dim, audio_dim,
                 use_txt_out=False, use_video=False, full_batch=True, use_moe=False, args=None):
        super(UniterThreeForPretrainingForT2Ifinetune, self).__init__()
        parallel_config = ParallelConfig()
        self.use_txt_out = use_txt_out
        self.use_video = use_video
        if self.use_video:
            img_dim = 1024

        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        config.use_vit = False
        config.seq_length = args.max_txt_len
        config.batch_size = args.train_batch_size
        self.uniter = UniterThreeModel(config, img_dim, audio_dim, use_video, parallel_config, use_moe)

        # T2I Generator
        group_for_loss = 2
        fusion_group_backbone = parallel_config.fusion_group
        parallel_config.fusion_group = group_for_loss
        parallel_config.optimizer_shard = False
        fg_backbone = fusion_group_backbone // parallel_config.fusion_group
        self.t2i_cfg = copy.deepcopy(cfg)
        self.t2i_cfg.seq_length = args.image_token_len + 100
        self.t2i_cfg.max_position_embeddings = args.image_token_len + 100
        self.t2i_cfg.vocab_size = args.image_codebook_size + 1
        self.t2i_cfg.max_decode_length = args.image_token_len + 1
        self.t2i_cfg.hidden_size = args.decoder_hidden_size
        self.t2i_cfg.num_hidden_layers = args.decoder_num_layers
        self.t2i_cfg.num_attention_heads = args.decoder_attn_heads

        self.t2i_output = TransformerModel(self.t2i_cfg, True, config.hidden_size, parallel_config, fg_backbone, False)
        self.id_crit = TransformerTrainingLoss(self.t2i_cfg, parallel_config)

    def generate_img(self, sequence_output, att_masks, img_gts, img_masks):
        # generate text
        t2i_out = self.t2i_output(sequence_output, att_masks, img_gts, img_masks)
        loss = self.id_crit(t2i_out, img_gts, img_masks)
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
        sequence_output, _ = self.uniter(input_ids, position_ids,
                                         None, None,
                                         attention_mask, gather_index,
                                         output_all_encoded_layers=False,
                                         txt_type_ids=None, img_type_ids=None,
                                         audio_feat=None, audio_pos_ids=None,
                                         audio_type_ids=None)

        id_loss = self.generate_img(sequence_output, attention_mask, txt_gts, txt_masks)

        return id_loss

class UniterThreeForPretrainingForT2IfinetuneInf(nn.Cell):
    """ UNITER pretraining """

    def __init__(self, config, img_dim, img_label_dim, audio_dim, audio_label_dim,
                 use_txt_out=False, use_video=False, use_vit=False, full_batch=True, use_moe=False, args=None):
        super(UniterThreeForPretrainingForT2IfinetuneInf, self).__init__()
        parallel_config = ParallelConfig()
        self.use_txt_out = use_txt_out
        self.use_video = use_video
        if self.use_video:
            img_dim = 1024

        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        config.use_vit = False
        config.seq_length = args.max_txt_len
        config.batch_size = args.val_batch_size
        self.uniter = UniterThreeModel(config, img_dim, audio_dim, use_video, parallel_config, use_moe)
        # T2I Generator

        group_for_loss = 2
        fusion_group_backbone = parallel_config.fusion_group
        parallel_config.fusion_group = group_for_loss
        parallel_config.optimizer_shard = False
        fg_backbone = fusion_group_backbone // parallel_config.fusion_group
        self.t2i_cfg = copy.deepcopy(cfg)
        self.t2i_cfg.seq_length = args.image_token_len + 100
        self.t2i_cfg.max_position_embeddings = args.image_token_len + 100
        self.t2i_cfg.vocab_size = args.image_codebook_size + 1
        self.t2i_cfg.max_decode_length = args.image_token_len + 1
        self.t2i_cfg.hidden_size = args.decoder_hidden_size
        self.t2i_cfg.num_hidden_layers = args.decoder_num_layers
        self.t2i_cfg.num_attention_heads = args.decoder_attn_heads
        self.t2i_cfg.batch_size = 1

        self.t2i_output = TransformerModel(self.t2i_cfg, False, config.hidden_size, parallel_config, fg_backbone, False, task="T2I")
        self.id_crit = TransformerTrainingLoss(self.t2i_cfg, parallel_config)

    def generate_img_eval(self, sequence_output, att_masks):
        # generate text
        t2i_out = self.t2i_output(sequence_output, att_masks)
        return t2i_out

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
                                         None, None,
                                         attention_mask, gather_index,
                                         output_all_encoded_layers=False,
                                         txt_type_ids=None, img_type_ids=None,
                                         audio_feat=None, audio_pos_ids=None,
                                         audio_type_ids=None)

        out = self.generate_img_eval(sequence_output, attention_mask)

        return out

