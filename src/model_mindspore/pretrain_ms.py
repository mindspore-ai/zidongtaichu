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
import yaml

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor

from src.model_mindspore.parallel_transformer import ParallelConfig
from src.model_mindspore.model_config import UniterConfig
from src.model_mindspore.transformer_model import TransformerModel
from src.model_mindspore.model_ms import UniterThreeModel, UniterThreeModelAudio
from src.model_mindspore.layer_ms import (BertOnlyMLMHead, RegionClassification, RegionFeatureRegression,
                                      AudioFeatureRegression)
from src.model_mindspore.loss import ContrastiveLoss, CrossEntropy, MSE, TransformerTrainingLoss
from src.config import config as C
from src.config.transformer_config import transformer_net_cfg as cfg

#from fastspeech2_ms.model.fastspeech2 import FastSpeech2ThreeV3
#from fastspeech2_ms.model.loss import FastSpeech2ThreeV3Loss


# Masked Language Modeling (MLM);
# Masked Region Feature Regression (MRFR);
# Masked Region Classification (MRC);
# Masked Region Classification with KL-divergence (MRC-kl);
# Image-Text Matching (ITM).
# Masked Audio Feature Regression (MAFR)
# Masked Audio Contrastive (MAC)
class UniterThreeForPretrainingWithLoss(nn.Cell):
    """ UNITER pretraining """

    def __init__(self, config, img_dim, img_label_dim, audio_dim, audio_label_dim,
                 use_txt_out=False, use_video=False, full_batch=True, use_moe=False, is_parallel=True, args=None):
        super(UniterThreeForPretrainingWithLoss, self).__init__()
        parallel_config = ParallelConfig()
        self.use_txt_out = use_txt_out
        self.use_video = use_video
        if self.use_video:
            img_dim = 1024
        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        self.uniter = UniterThreeModel(config, img_dim, audio_dim, use_video, parallel_config, use_moe, is_parallel)

        self.cls = BertOnlyMLMHead(config, self.uniter.embeddings.word_embeddings.embedding_table, parallel_config)

        self.region_classifier = RegionClassification(config, config.hidden_size, img_label_dim, parallel_config)

        self.feat_regress = RegionFeatureRegression(config, config.hidden_size, img_dim,
                                                    self.uniter.img_embeddings.img_linear.weight, parallel_config)

        self.audio_feat_regress = AudioFeatureRegression(config, config.hidden_size, audio_dim,
                                                         self.uniter.audio_embeddings.audio_linear.weight,
                                                         parallel_config)

        self.itm_output = nn.Dense(config.hidden_size, 5).to_float(mindspore.float16)
        self.itm_output.matmul.shard(((parallel_config.dp, 1), (1, 1)))
        self.itm_output.bias_add.shard(((parallel_config.dp, 1), (1,)))
        self.itm_output.weight.parallel_optimizer = False
        self.itm_output.bias.parallel_optimizer = False

        # Text Generator
        group_for_loss = 2
        fusion_group_backbone = parallel_config.fusion_group
        parallel_config.fusion_group = group_for_loss
        parallel_config.optimizer_shard = False
        fg_backbone = fusion_group_backbone // parallel_config.fusion_group
        txt_cfg = copy.deepcopy(cfg)
        self.txt_output = TransformerModel(txt_cfg, True, config.hidden_size, parallel_config, fg_backbone, False)
        self.td_crit = TransformerTrainingLoss(txt_cfg, parallel_config)

        # Image Generator
        img_cfg = copy.deepcopy(cfg)
        img_cfg.vocab_size = C.IMG_TOKEN_SIZE
        img_cfg.seq_length = C.IMG_TOKEN_LEN
        self.img_output = TransformerModel(img_cfg, True, config.hidden_size, parallel_config, fg_backbone + 2, False)
        self.id_crit = TransformerTrainingLoss(img_cfg, parallel_config)

        # Audio Generaotr
        # preprocess_config = yaml.load(open(args.audio_preprocess_config, "r"), Loader=yaml.FullLoader)
        # model_config = yaml.load(open(args.audio_model_config, "r"), Loader=yaml.FullLoader)
        # self.audio_output = FastSpeech2Three(preprocess_config, model_config, config.hidden_size)
        # self.audio_crit = FastSpeech2ThreeLoss()

        self.logit_temp = 0.1  # temperature to divide logits by
        self.n_negatives = 10  # number of negative examples from the same sample
        self.cross_sample_negatives = 0  # number of negative examples from the any sample

        self.cross_entropy = CrossEntropy(parallel_config)
        self.mse_loss = MSE(parallel_config)
        self.contrastive_loss = ContrastiveLoss(self.logit_temp, parallel_config)
        self.concat = ops.Concat(axis=0).shard(((1,), (1,), (1,), (1,), (1,), (1,), (1,)))
        self.mul = ops.Mul().shard(((1,), (1, 1)))
        self.expand_dims = ops.ExpandDims()
        self.squeeze_1 = ops.Squeeze(1).shard(((parallel_config.dp, 1),))
        self.gather_nd = ops.GatherNd().shard(((1, 1, 1), (1, 1)))
        self.tile = ops.Tile()
        self.cast = ops.Cast()
        self.one_hot = ops.OneHot().shard(((1, 1), (), ()))
        self.on_value = Tensor(1.0, mindspore.float32)
        self.off_value = Tensor(0.0, mindspore.float32)
        self.reduce_sum = ops.ReduceSum().shard(((1, 1),))
        self.slice = ops.StridedSlice().shard(((parallel_config.dp, 1, 1),))
        self.add = ops.Add().shard(((), ()))
        self.full_batch = full_batch
        self.stride_slice_1 = ops.StridedSlice().shard(((1,),))
        self.stride_slice_2 = ops.StridedSlice().shard(((1, 1),))

        self.mean = ops.ReduceMean().shard(((1,),))

    def generate_text(self, sequence_output, att_masks, txt_gts, txt_masks):
        """Generate text"""
        txt_out = self.txt_output(sequence_output, att_masks, txt_gts, txt_masks)
        loss = self.td_crit(txt_out, txt_gts, txt_masks)
        return loss

    def generate_img(self, sequence_output, att_masks, img_gts, img_masks):
        """Generate img"""
        img_out = self.img_output(sequence_output, att_masks, img_gts, img_masks)
        loss = self.id_crit(img_out, img_gts, img_masks)
        return loss

    # def generate_audio(self, sequence_output, mel_targets, src_masks, mel_masks, duration_targets):
    #
    #     # mel_targets, src_masks, mel_masks, duration_targets, mel_predictions, postnet_mel_predictions, log_duration_predictions
    #
    #     input_data = sequence_output
    #
    #     mel_predictions, postnet_mel_predictions, log_duration_predictions, d_rounded  = self.audio_output(input_data, src_masks, mel_masks, duration_targets)
    #
    #     # mel_targets, src_masks, mel_masks, duration_targets, mel_predictions, postnet_mel_predictions, log_duration_predictions
    #     loss = self.audio_crit(mel_targets, src_masks, mel_masks, duration_targets,
    #                            mel_predictions, postnet_mel_predictions, log_duration_predictions)
    #
    #     return loss

    def forward_td_three(self, sequence_output, attention_mask, txt_gts, txt_masks):
        """Forward function for td_three"""
        td_loss = self.generate_text(sequence_output, attention_mask, txt_gts, txt_masks)
        return td_loss

    def forward_id_three(self, sequence_output, attention_mask, img_token_gts, img_token_masks):
        """Forward function for id_three"""
        id_loss = self.generate_img(sequence_output, attention_mask, img_token_gts, img_token_masks)
        return id_loss

    # def forward_ad_three(self, sequence_output, mel_targets, src_masks, mel_masks, duration_targets):
    #     ad_loss = self.generate_audio(sequence_output,mel_targets, src_masks, mel_masks, duration_targets)
    #     return ad_loss

    def forward_mlm_three(self, sequence_output, input_ids, txt_mask, txt_label_mask):
        """Forward function for mlm_three"""
        # get only the text part
        # sequence_output1 = sequence_output[:, :input_ids.shape[1], :]
        sequence_output1 = self.slice(sequence_output, (0, 0, 0),
                                      (sequence_output.shape[0], input_ids.shape[1], sequence_output.shape[2]),
                                      (1, 1, 1))
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output1, txt_mask)
        prediction_scores = self.cls(masked_output)

        masked_lm_loss = self.cross_entropy(prediction_scores, txt_label_mask)

        return masked_lm_loss

    def forward_mrc_three(self, sequence_output, img_mask_tgt_mask, label_targets):
        """Forward function for mrc_three"""
        # only compute masked regions for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt_mask)

        # prediction_soft_label: 412, 1601
        prediction_soft_label = self.region_classifier(masked_output)

        label_targets = self.squeeze_1(label_targets)

        label_targets = self.cast(label_targets, mstype.int32)
        mrc_loss = self.cross_entropy(prediction_soft_label, label_targets)

        return mrc_loss

    def forward_mrfr_three(self, sequence_output, img_mask_tgt_mask, ori_img_feat, img_masks):
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output, img_mask_tgt_mask)
        feat_targets = self._compute_masked_hidden(ori_img_feat, img_masks)
        prediction_feat = self.feat_regress(masked_output)

        mrfr_loss = self.mse_loss(prediction_feat, feat_targets)
        return mrfr_loss

    def forward_mrct_three(self, sequence_output, img_mask_tgt_mask, ori_img_feat, img_masks, neg_index, neg_sample):
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output, img_mask_tgt_mask)
        feat_targets = self._compute_masked_hidden(ori_img_feat, img_masks)

        prediction_feat = self.feat_regress(masked_output)
        mrct_loss = self.contrastive_loss(prediction_feat, feat_targets, neg_sample)
        return mrct_loss

    def forward_mafr_three(self, sequence_output, audio_mask_tgt_mask, feat_targets):
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output, audio_mask_tgt_mask)
        prediction_feat = self.audio_feat_regress(masked_output)

        mafr_loss = self.mse_loss(prediction_feat, feat_targets)
        return mafr_loss

    def forward_mac_three(self, sequence_output, audio_masks, audio_mask_tgt, feat_targets, ma_neg_index,
                          ma_neg_sample):
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output, audio_mask_tgt)

        prediction_feat = self.audio_feat_regress(masked_output)

        mac_loss = self.contrastive_loss(prediction_feat, feat_targets, ma_neg_sample)

        # mac_loss = mac_loss/ma_neg_index.shape[0]
        return mac_loss

    def forward_itm_three(self, sequence_output, targets):
        """Forward function for itm_three"""
        pooled_output = self.uniter.pooler(sequence_output)
        itm_scores = self.itm_output(pooled_output)
        itm_scores = itm_scores.astype(mindspore.float32)

        itm_loss = self.cross_entropy(itm_scores, targets)
        return itm_loss

    # hidden: 32 * 59 * 768
    # mask: 56, 155
    # hidden: 56, 155, 768
    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        hidden_masked = self.gather_nd(hidden, mask)

        return hidden_masked

    def construct(self, input_ids, position_ids, img_feat, img_pos_feat, audio_feat, audio_pos_ids,
                  attention_mask, gather_index, txt_labels, txt_mask, txt_label_mask, img_mask_tgt,
                  img_mask_tgt_mask, img_masks, mrc_label_target, mrfr_feat_target,
                  audio_mask_tgt, audio_masks, mafr_feat_target, itm_targets, ma_neg_index, ma_neg_sample,
                  mr_neg_index, mr_neg_sample, txt_gts, txt_masks, img_token_gts, img_token_masks, images, images_mask,
                  taskId):
        """Construct Function"""
        if not self.full_batch:
            taskId = self.stride_slice_1(taskId, (0,), (1,), (1,))
            position_ids = self.stride_slice_2(position_ids, (0, 0), (1, C.MAX_FULL_TEXT_LEN), (1, 1))
            audio_pos_ids = self.stride_slice_2(audio_pos_ids, (0, 0), (1, C.MAX_AUDIO_LEN), (1, 1))
        one_hot_task_id = self.one_hot(taskId.view(1, ), 7, self.on_value, self.off_value)
        sequence_output, moe_loss, ori_img_feat = self.uniter(input_ids, position_ids,
                                                              img_feat, img_pos_feat,
                                                              attention_mask, gather_index, img_masks=img_masks,
                                                              output_all_encoded_layers=False,
                                                              txt_type_ids=None, img_type_ids=None,
                                                              audio_feat=audio_feat, audio_pos_ids=audio_pos_ids,
                                                              audio_type_ids=None, audio_masks=audio_masks,
                                                              images=images)


        mlm_loss = self.forward_mlm_three(sequence_output, input_ids, txt_mask, txt_label_mask)
        mrfr_loss = self.forward_mrfr_three(sequence_output, img_mask_tgt_mask, ori_img_feat, images_mask)
        mafr_loss = self.forward_mafr_three(sequence_output, audio_mask_tgt, mafr_feat_target)
        mac_loss = self.forward_mac_three(sequence_output, audio_masks, audio_mask_tgt,
                                          mafr_feat_target, ma_neg_index, ma_neg_sample)
        itm_loss = self.forward_itm_three(sequence_output, itm_targets)
        td_loss = self.forward_td_three(sequence_output, attention_mask, txt_gts, txt_masks)
        id_loss = self.forward_id_three(sequence_output, attention_mask, img_token_gts, img_token_masks)
        # ad_loss = self.forward_ad_three(sequence_output, audio_mel_targets, audio_src_masks, audio_mel_masks, audio_duration_targets)
        loss = self.concat((mlm_loss.view(1, ), mafr_loss.view(1, ), mrfr_loss.view(1, ),
                            mac_loss.view(1, ), itm_loss.view(1, ), td_loss.view(1, ), id_loss.view(1, )))
        loss = self.mul(loss, one_hot_task_id)
        final_loss = self.reduce_sum(loss)
        return final_loss

