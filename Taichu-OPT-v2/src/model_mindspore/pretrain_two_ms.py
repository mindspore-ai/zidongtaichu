# Copyright 2021 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor

from src.config.model_config import UniterConfig
from src.model_mindspore.model_ms import UniterThreeModel, freeze_net
from src.model_mindspore.layer_ms import BertOnlyMLMHead
from src.model_mindspore.loss import CrossEntropy, ClipLoss
from src.config import config as C

import numpy as np
from mindspore.common.parameter import Parameter
from mindspore.communication.management import get_rank, get_group_size
from src.model_mindspore.parallel_transformer import LayerNorm, ParallelConfig

class UniterTwoForPretrainingWithLoss(nn.Cell):
    """ UNITER pretraining """

    def __init__(self, config, args):
        super(UniterTwoForPretrainingWithLoss, self).__init__()

        img_dim = C.IMG_DIM
        audio_dim = C.AUDIO_DIM
        full_batch = args.full_batch
        use_moe = args.use_moe
        is_parallel = args.use_pipeline

        parallel_config = ParallelConfig()
        if(isinstance(config, UniterConfig) == False):
            config = UniterConfig.from_json_file(config)
            config.full_batch = full_batch
            config.use_pipeline = args.use_pipeline
            config.batch_size = args.train_batch_size
            config.seq_length = C.MAX_IMG_TEXT_LEN
            config.patch_size = C.IMG_PATCH_SIZE
            config.train_image_size = C.IMG_SIZE

        self.uniter = UniterThreeModel(config, img_dim, audio_dim, parallel_config, use_moe, is_parallel)

        bert_type = getattr(config, "bert_type", 0)
        if bert_type == 0:
            self.cls = BertOnlyMLMHead(config, self.uniter.embeddings.word_embeddings.embedding_table, parallel_config)
        elif bert_type == 1:
            self.cls = BertOnlyMLMHead(config, self.uniter.embeddings.embeddings.word_embeddings.embedding_table, parallel_config)

        self.itm_output = nn.Dense(config.hidden_size, 2).to_float(mindspore.float16)
        self.itm_output.matmul.shard(((parallel_config.dp, 1), (1, 1)))
        self.itm_output.bias_add.shard(((parallel_config.dp, 1), (1,)))
        self.itm_output.weight.parallel_optimizer = False
        self.itm_output.bias.parallel_optimizer = False

        if config.use_pipeline:
            self.cls.pipeline_stage = 7
            self.itm_output.pipeline_stage = 7

        self.logit_temp = 0.1  # temperature to divide logits by
        self.n_negatives = 10  # number of negative examples from the same sample
        self.cross_sample_negatives = 0  # number of negative examples from the any sample

        self.cross_entropy = CrossEntropy(parallel_config)
        self.concat = ops.Concat(axis=0).shard(((1,), (1,), (1,)))
        self.mul = ops.Mul().shard(((1,), (1, 1)))
        self.gather_nd = ops.GatherNd().shard(((1, 1, 1), (1, 1)))
        self.one_hot = ops.OneHot().shard(((1, 1), (), ()))
        self.on_value = Tensor(1.0, mindspore.float32)
        self.off_value = Tensor(0.0, mindspore.float32)
        self.reduce_sum = ops.ReduceSum().shard(((1, 1),))
        self.slice = ops.StridedSlice().shard(((parallel_config.dp, 1, 1),))
        self.add = ops.Add().shard(((), ()))
        self.full_batch = full_batch
        self.stride_slice_1 = ops.StridedSlice().shard(((1,),))
        self.stride_slice_2 = ops.StridedSlice().shard(((1, 1),))

        self.clip_temp = Parameter(Tensor(0.07, mstype.float32))
        self.div = ops.Div().shard(((1, 1), (1, 1)))
        self.norm = nn.Norm(axis=-1, keep_dims=True)
        self.bmm = ops.MatMul(transpose_a=False, transpose_b=True)
        self.transpose = ops.Transpose()

        self.cat = ops.Concat(axis=0).shard(((1, 1), (1, 1)))
        self.eq = ops.Equal()
        self.softmax = ops.Softmax(axis=1)
        self.ones = ops.Ones()
        self.zeros = ops.Zeros()
        self.stack = ops.Stack(axis=0)
        self.cat_dim1 = ops.Concat(axis=1)

        self.text_proj = nn.Dense(config.hidden_size, args.embed_size).to_float(mstype.float16)
        self.vision_proj = nn.Dense(config.hidden_size, args.embed_size).to_float(mstype.float16)

        self.temp = 0.07
        self.normalize = ops.L2Normalize(axis=-1)
        self.exp = ops.Exp()
        self.clip_loss = ClipLoss(config.batch_size, parallel_config, self.temp)

        group_size = 1
        try:
            self.allgather = ops.AllGather()
            self.rank = get_rank()
            group_size = get_group_size()
        except Exception as e:
            print(e)

        self.itm_head_two = nn.SequentialCell(
                nn.Dense(config.hidden_size, config.hidden_size * 2).to_float(mstype.float16),
                LayerNorm((config.hidden_size * 2,), parallel_config.dp),
                nn.GELU(),
                nn.Dense(config.hidden_size * 2, 2).to_float(mstype.float16)
            )

        batch_size = config.batch_size

        mask_diagonal = np.zeros((batch_size, batch_size))
        np.fill_diagonal(mask_diagonal, 1)
        self.mask_diagonal = Tensor(mask_diagonal, mstype.bool_)

        global_batch_size = batch_size * group_size
        global_mask_diagonal = np.zeros((global_batch_size, global_batch_size))
        np.fill_diagonal(global_mask_diagonal, 1)
        self.global_mask_diagonal = Tensor(global_mask_diagonal, mstype.bool_)

        self.fill_diagonal = ops.MaskedFill()

        self.itm_targets = Tensor(np.concatenate([np.ones(batch_size, np.int32),
                                           np.zeros(batch_size*2, np.int32)]))

        self.topk = ops.TopK()
        self.argmax = ops.Argmax()
        self.mean = ops.ReduceMean()

        self.sort = ops.Sort(axis=-1, descending=False)
        self.sortD = ops.Sort(axis=-1, descending=True)

        self.gather_full = Tensor(np.arange(0, C.MAX_IMG_TEXT_LEN), mstype.int32)

        self.img_emb_type = getattr(args, 'img_emb_type', 0)

        self.batch_size = batch_size

        if getattr(args, 'freeze_clip', False):
            print("freeze_clip True")
            freeze_net(self.vision_proj)
            freeze_net(self.text_proj)
            freeze_net(self.clip_loss)
        else:
            print("freeze_clip False")


        if getattr(args, 'use_global_neg', False):
            print("use_global_neg True")
            self.get_neg = self.get_global_neg_feat
        else:
            print("use_global_neg False")
            self.get_neg = self.get_neg_feat

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


    def forward_itm_three(self, sequence_output, targets):
        """Forward function for itm_three"""
        pooled_output = self.uniter.get_first_token(sequence_output)
        itm_scores = self.itm_head_two(pooled_output)
        itm_scores = itm_scores.astype(mindspore.float32)

        itm_loss = self.cross_entropy(itm_scores, targets)
        return itm_loss

    # hidden: 32 * 59 * 768"""  """
    # mask: 56, 155
    # hidden: 56, 155, 768
    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        hidden_masked = self.gather_nd(hidden, mask)

        return hidden_masked


    def get_txt_feat(self, input_ids, position_ids, attn_masks_text):
        txt_emb = self.uniter._compute_txt_embeddings(input_ids, position_ids, attn_masks_text=attn_masks_text)
        return txt_emb

    def get_img_feat(self, images):
        img_emb = self.uniter.img_embeddings.get_feat(images)
        return img_emb


    def get_neg_feat(self, input_ids, images,
                     pool_feat_image, pool_feat_text,
                     attn_masks_text, attn_masks_img):

        sim_i2t = self.bmm(pool_feat_image.astype(mstype.float16), pool_feat_text.astype(mstype.float16))
        sim_i2t = sim_i2t * self.exp(self.clip_loss.clip_temp)

        weights_i2t = self.softmax(sim_i2t) + 1e-5
        weights_i2t = self.fill_diagonal(weights_i2t, self.mask_diagonal, 0.0)

        neg_idx_text = self.argmax(weights_i2t)
        input_ids_neg = input_ids[neg_idx_text]
        txt_att_neg = attn_masks_text[neg_idx_text]

        attn_mask_neg_txt = self.cat_dim1([attn_masks_img, txt_att_neg])

        sim_t2i = sim_i2t.transpose()
        weights_t2i = self.softmax(sim_t2i) + 1e-5
        weights_t2i = self.fill_diagonal(weights_t2i, self.mask_diagonal, 0.0)

        neg_idx_img = self.argmax(weights_t2i)
        images_neg = images[neg_idx_img]

        return (input_ids_neg, attn_mask_neg_txt, txt_att_neg, images_neg)

    def get_global_neg_feat(self, input_ids, images,
                     pool_feat_image, pool_feat_text,
                     attn_masks_text, attn_masks_img):

        input_ids = self.allgather(input_ids)
        images = self.allgather(images)
        pool_feat_image = self.allgather(pool_feat_image)
        pool_feat_text = self.allgather(pool_feat_text)
        attn_masks_text = self.allgather(attn_masks_text)
        attn_masks_img = self.allgather(attn_masks_img)

        sim_i2t = self.bmm(pool_feat_image.astype(mstype.float16), pool_feat_text.astype(mstype.float16))
        sim_i2t = sim_i2t * self.exp(self.clip_loss.clip_temp)

        weights_i2t = self.softmax(sim_i2t) + 1e-5
        weights_i2t = self.fill_diagonal(weights_i2t, self.global_mask_diagonal, 0.0)

        neg_idx_text = self.argmax(weights_i2t)
        input_ids_neg = input_ids[neg_idx_text]
        txt_att_neg = attn_masks_text[neg_idx_text]

        attn_mask_neg_txt = self.cat_dim1([attn_masks_img, txt_att_neg])

        sim_t2i = sim_i2t.transpose()
        weights_t2i = self.softmax(sim_t2i) + 1e-5
        weights_t2i = self.fill_diagonal(weights_t2i, self.global_mask_diagonal, 0.0)

        neg_idx_img = self.argmax(weights_t2i)
        images_neg = images[neg_idx_img]

        input_ids_neg = input_ids_neg[self.rank*self.batch_size: (self.rank+1)*self.batch_size]
        attn_mask_neg_txt = attn_mask_neg_txt[self.rank*self.batch_size: (self.rank+1)*self.batch_size]
        txt_att_neg = txt_att_neg[self.rank*self.batch_size: (self.rank+1)*self.batch_size]
        images_neg = images_neg[self.rank*self.batch_size: (self.rank+1)*self.batch_size]

        return (input_ids_neg, attn_mask_neg_txt, txt_att_neg, images_neg)


    def forward_itmHard_three(self, input_ids, position_ids, images,
                              attn_masks_text, attn_masks_img,
                              attention_mask,
                              image_feat, text_feat,
                              pool_feat_text, pool_feat_image):

        (input_ids_neg, attn_mask_neg_txt,
         txt_att_neg, images_neg) = self.get_neg(input_ids, images,
                                               pool_feat_image,
                                               pool_feat_text,
                                               attn_masks_text,
                                               attn_masks_img)

        itm_loss = self.forward_itmHard_loss(text_feat, image_feat,
                             attention_mask, attn_masks_text,
                             input_ids_neg, position_ids, txt_att_neg,
                             attn_mask_neg_txt, images_neg)

        return itm_loss


    def forward_itmHard_local_three(self, input_ids, position_ids, images,
                              attn_masks_text, attn_masks_img,
                              attention_mask,
                              image_feat, text_feat,
                              pool_feat_text, pool_feat_image):

        (input_ids_neg, attn_mask_neg_txt,
         txt_att_neg, images_neg) = self.get_neg_feat(input_ids, images,
                                               pool_feat_image,
                                               pool_feat_text,
                                               attn_masks_text,
                                               attn_masks_img)

        itm_loss = self.forward_itmHard_loss(text_feat, image_feat,
                             attention_mask, attn_masks_text,
                             input_ids_neg, position_ids, txt_att_neg,
                             attn_mask_neg_txt, images_neg)

        return itm_loss


    def forward_itmHard_global_three(self, input_ids, position_ids, images,
                              attn_masks_text, attn_masks_img,
                              attention_mask,
                              image_feat, text_feat,
                              pool_feat_text, pool_feat_image):

        (input_ids_neg, attn_mask_neg_txt,
         txt_att_neg, images_neg) = self.get_global_neg_feat(input_ids, images,
                                               pool_feat_image,
                                               pool_feat_text,
                                               attn_masks_text,
                                               attn_masks_img)

        itm_loss = self.forward_itmHard_loss(text_feat, image_feat,
                             attention_mask, attn_masks_text,
                             input_ids_neg, position_ids, txt_att_neg,
                             attn_mask_neg_txt, images_neg)

        return itm_loss

    def forward_itmHard_loss(self, text_feat, image_feat,
                             attention_mask, attn_masks_text,
                             input_ids_neg, position_ids, txt_att_neg,
                             attn_mask_neg_txt, images_neg):

        output_pos, _ = self.uniter.feat_fusion(text_feat, image_feat,
                                                attention_mask, attn_masks_text)
        feat_pos = self.uniter.get_first_token(output_pos)

        text_feat_neg = self.get_txt_feat(input_ids_neg, position_ids, txt_att_neg)
        output_neg_txt, _ = self.uniter.feat_fusion(text_feat_neg, image_feat,
                                                    attn_mask_neg_txt, txt_att_neg)
        feat_neg_txt = self.uniter.get_first_token(output_neg_txt)

        image_feat_neg = self.get_img_feat(images_neg)
        output_neg_img, _ = self.uniter.feat_fusion(text_feat, image_feat_neg,
                                                    attention_mask, attn_masks_text)
        feat_neg_img = self.uniter.get_first_token(output_neg_img)

        feat_total = self.cat([feat_pos, feat_neg_txt, feat_neg_img])

        itm_scores = self.itm_head_two(feat_total)

        itm_loss = self.cross_entropy(itm_scores, self.itm_targets)

        return itm_loss


    def get_text_image_feat(self, text_embed, image_embed):

        feat_text = self.text_proj(self.uniter.get_first_token(text_embed))
        feat_text = self.normalize(feat_text)

        feat_image = self.vision_proj(self.uniter.get_first_token(image_embed))
        feat_image = self.normalize(feat_image)

        return feat_text, feat_image

    def construct(self, input_ids, position_ids, attention_mask,
                  txt_mask, txt_label_mask, itm_target,
                  attn_masks_text, attn_masks_img,
                  images, images_rand, input_ids_mask, taskId):

        """Construct Function"""
        if not self.full_batch:
            taskId = self.stride_slice_1(taskId, (0,), (1,), (1,))
            position_ids = self.stride_slice_2(position_ids, (0, 0), (1, C.MAX_FULL_TEXT_LEN), (1, 1))

        text_embed_mask = self.get_txt_feat(input_ids_mask, position_ids, attn_masks_text)
        image_embed = self.get_img_feat(images)

        sequence_output_mask, moe_loss = self.uniter.feat_fusion(text_embed_mask, image_embed, attention_mask, attn_masks_text)
        mlm_loss = self.forward_mlm_three(sequence_output_mask, input_ids_mask, txt_mask, txt_label_mask)

        text_embed = self.get_txt_feat(input_ids, position_ids, attn_masks_text)
        pool_feat_text, pool_feat_image = self.get_text_image_feat(text_embed, image_embed)

        itc_loss = self.clip_loss(pool_feat_text, pool_feat_image)

        itm_loss = self.forward_itmHard_three(input_ids, position_ids, images,
                                              attn_masks_text, attn_masks_img,
                                              attention_mask,
                                              image_embed, text_embed,
                                              pool_feat_text, pool_feat_image)

        loss = self.concat((mlm_loss.view(1, ), itc_loss.view(1, ), itm_loss.view(1, )))
        final_loss = self.reduce_sum(loss)

        print(f"mlm_loss {self.mean(mlm_loss.view(1, ))}")
        print(f"itc_loss {self.mean(itc_loss.view(1, ))}")
        print(f"itm_loss {self.mean(itm_loss.view(1, ))}")

        return final_loss
