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
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor

from src.model_mindspore.model_config import UniterConfig
from src.config import config as C
from src.model_mindspore.parallel_transformer import ParallelConfig
from src.model_mindspore.model_ms import UniterThreeModel
class UniterThreeForPretrainingForRetFinetune(nn.Cell):
    """ UNITER pretraining """

    def __init__(self, config, img_dim, img_label_dim, audio_dim, audio_label_dim,
                 use_txt_out=False, use_video=False, full_batch=True, use_moe=False, is_parallel=True, args=None, margin=0.2):
        super(UniterThreeForPretrainingForRetFinetune, self).__init__()
        parallel_config = ParallelConfig()
        self.is_parallel = is_parallel
        self.use_txt_out = use_txt_out
        self.use_video = use_video
        if self.use_video:
            img_dim = 1024
        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        self.uniter = UniterThreeModel(config, img_dim, audio_dim, use_video, parallel_config, use_moe, is_parallel)

        self.itm_output = nn.Dense(config.hidden_size, 5).to_float(mindspore.float16)
        self.itm_output.matmul.shard(((parallel_config.dp, 1), (1, 1)))
        self.itm_output.bias_add.shard(((parallel_config.dp, 1), (1,)))
        self.itm_output.weight.parallel_optimizer = False
        self.itm_output.bias.parallel_optimizer = False

        self.full_batch = full_batch
        self.stride_slice_1 = ops.StridedSlice().shard(((1,),))
        self.stride_slice_2 = ops.StridedSlice().shard(((1, 1),))

        self.mean = ops.ReduceMean().shard(((1,),))

        self.rank_output = nn.Dense(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.margin = margin
        self.min_value = Tensor(0, mindspore.float32)
        self.max_value = Tensor(100, mindspore.float32)
        self.print = ops.Print()
        if self.is_parallel:
            self.allgather = ops.AllGather()

    def init_output(self):
        self.rank_output.weight.set_data(self.itm_output.weight.data[2:3, :])
        self.rank_output.bias.set_data(self.itm_output.bias.data[2:3])

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
        sequence_output, moe_loss, ori_img_feat = self.uniter(input_ids, position_ids,
                                                              img_feat, img_pos_feat,
                                                              attention_mask, gather_index, img_masks=img_masks,
                                                              output_all_encoded_layers=False,
                                                              txt_type_ids=None, img_type_ids=None,
                                                              audio_feat=audio_feat, audio_pos_ids=audio_pos_ids,
                                                              audio_type_ids=None, audio_masks=audio_masks,
                                                              images=images)

        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores = self.rank_output(pooled_output)
        rank_scores_sigmoid = self.sigmoid(rank_scores)
        sample_size = 3  # 2*neg_samples+1
        scores = rank_scores_sigmoid.view(-1, sample_size)
        pos = self.stride_slice_2(scores, (0, 0), (scores.shape[0], 1), (1, 1))
        neg = self.stride_slice_2(scores, (0, 1), (scores.shape[0], scores.shape[1]), (1, 1))
        mat = self.margin + neg - pos
        rank_loss = ops.clip_by_value(mat, self.min_value, self.max_value)
        rank_loss = self.allgather(rank_loss)
        rank_loss_mean = rank_loss.mean()

        return rank_loss_mean

class UniterThreeForPretrainingForRetFinetuneEval(nn.Cell):
    """ UNITER pretraining ret ft eval """

    def __init__(self, config, img_dim, img_label_dim, audio_dim, audio_label_dim,
                 use_txt_out=False, use_video=False, full_batch=True, use_moe=False, is_parallel=True, args=None, margin=0.2):
        super(UniterThreeForPretrainingForRetFinetuneEval, self).__init__()
        parallel_config = ParallelConfig()
        self.is_parallel = is_parallel
        self.use_txt_out = use_txt_out
        self.use_video = use_video
        if self.use_video:
            img_dim = 1024
        config = UniterConfig.from_json_file(config)
        config.full_batch = full_batch
        self.uniter = UniterThreeModel(config, img_dim, audio_dim, use_video, parallel_config, use_moe, is_parallel)

        self.itm_output = nn.Dense(config.hidden_size, 5).to_float(mindspore.float16)
        self.itm_output.matmul.shard(((parallel_config.dp, 1), (1, 1)))
        self.itm_output.bias_add.shard(((parallel_config.dp, 1), (1,)))
        self.itm_output.weight.parallel_optimizer = False
        self.itm_output.bias.parallel_optimizer = False

        self.full_batch = full_batch
        self.stride_slice_1 = ops.StridedSlice().shard(((1,),))
        self.stride_slice_2 = ops.StridedSlice().shard(((1, 1),))
        self.rank_output = nn.Dense(config.hidden_size, 1)

        self.is_parallel = is_parallel
        if self.is_parallel:
            self.allgather = ops.AllGather()
    def init_output(self):
        self.rank_output.weight.set_data(self.itm_output.weight.data[2:3, :])
        self.rank_output.bias.set_data(self.itm_output.bias.data[2:3])

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
        sequence_output, moe_loss, ori_img_feat = self.uniter(input_ids, position_ids,
                                                              img_feat, img_pos_feat,
                                                              attention_mask, gather_index, img_masks=img_masks,
                                                              output_all_encoded_layers=False,
                                                              txt_type_ids=None, img_type_ids=None,
                                                              audio_feat=audio_feat, audio_pos_ids=audio_pos_ids,
                                                              audio_type_ids=None, audio_masks=audio_masks,
                                                              images=images)
        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores = self.rank_output(pooled_output)
        if self.is_parallel:
            rank_scores = self.allgather(rank_scores)
        return rank_scores
