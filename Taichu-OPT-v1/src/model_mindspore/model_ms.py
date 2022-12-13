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
""" model_ms """

import logging
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.parallel.nn import TransformerEncoder
from mindspore.nn import Cell
from mindspore.nn.transformer import TransformerOpParallelConfig
from src.config import config as C
from src.model_mindspore.parallel_transformer import Dropout, LayerNorm, ParallelConfig
from src.mae_mindspore.src.vit import VitEval, VitEvalV2

logger = logging.getLogger(__name__)


class BertAttentionMask(Cell):
    r"""
    Get the Lower triangular matrix.
    Args:
        seq_length: the length of the
        config(parallel_config): the parallel configure
    Inputs:
        input_mask: the mask indicating whether each position is a valid input with (batch_size, seq_length)
    Outputs:
        attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, parallel_config=ParallelConfig):
        super(BertAttentionMask, self).__init__()
        self.reshape = P.Reshape()
        self.mul = P.BatchMatMul().shard(
            ((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.expand_dim = P.ExpandDims().shard(((1, 1),))
        #         self.expand_dim_3 = P.ExpandDims().shard(((parallel_config.dp, 1, 1),))
        # Default lower triangle mask matrix
        self.multiply = P.Mul().shard(((parallel_config.dp, 1, 1), (1, 1, 1)))

    def construct(self, input_mask):
        r"""
        Generate the attention mask matrix.
        """
        input_shape = P.Shape()(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        # Mask the padded inputs
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.mul(mask_left, mask_right)
        # [bs, seq_length, seq_length]
        return attention_mask


#
class BertPooler(nn.Cell):
    """ BertPooler """

    def __init__(self, config, parallel_config):
        super(BertPooler, self).__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size).to_float(mstype.float16)
        self.dense.matmul.shard(((parallel_config.dp, 1), (1, 1)))
        self.dense.bias_add.shard(((parallel_config.dp, 1), (1,)))
        self.dense.weight.parallel_optimizer = False
        self.dense.bias.parallel_optimizer = False
        self.activation = nn.Tanh()
        self.activation.tanh.shard(((parallel_config.dp, 1),))

        self.slice = P.StridedSlice().shard(((parallel_config.dp, 1, 1),))

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = self.slice(hidden_states, (0, 0, 0), (hidden_states.shape[0], 1, hidden_states.shape[2]),
                                        (1, 1, 1))
        pooled_output = self.dense(first_token_tensor.view(hidden_states.shape[0], hidden_states.shape[2]))
        pooled_output = self.activation(pooled_output)
        return pooled_output.astype(mstype.float32)


class UniterEncoder(nn.Cell):
    """ UniterEncoder """

    def __init__(self, config, parallel_config, use_moe, is_parallel):
        super().__init__()
        self.num_layers = config.num_hidden_layers

        self.attention_mask = BertAttentionMask(parallel_config)

        op_parallel_config = TransformerOpParallelConfig(data_parallel=parallel_config.dp,
                                                         model_parallel=parallel_config.mp,
                                                         pipeline_stage=parallel_config.pipeline_stage,
                                                         optimizer_shard=parallel_config.optimizer_shard,
                                                         vocab_emb_dp=parallel_config.vocab_emb_dp)

        self.encoder = TransformerEncoder(batch_size=config.batch_size,
                                          num_layers=config.num_hidden_layers,
                                          hidden_size=config.hidden_size,
                                          ffn_hidden_size=config.intermediate_size,
                                          num_heads=config.num_attention_heads,
                                          seq_length=config.seq_length,
                                          attention_dropout_rate=config.attention_probs_dropout_prob,
                                          hidden_dropout_rate=config.hidden_dropout_prob,
                                          hidden_act=config.hidden_act,
                                          post_layernorm_residual=False,
                                          parallel_config=op_parallel_config)

    def construct(self, hidden_states, attention_mask):
        """ construct """

        attention_mask = self.attention_mask(attention_mask)
        # for i in range(self.num_layers):
        #     hidden_states, _ = self.encoder_layers[i](hidden_states, attention_mask)  #bs, seq_length, hidden_size  dp, 1, 1

        hidden_states, past = self.encoder(hidden_states, attention_mask)

        return hidden_states, past


class UniterTextEmbeddings(nn.Cell):
    """ UniterTextEmbeddings """

    def __init__(self, config, parallel_config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        self.word_embeddings.gather.shard(((1, 1), (parallel_config.dp,)))
        self.word_embeddings.expand.shard(((parallel_config.dp, 1),))
        self.word_embeddings.embedding_table.parallel_optimizer = False

        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.position_embeddings.gather.shard(((1, 1), (1,)))
        self.position_embeddings.expand.shard(((1, 1),))
        self.position_embeddings.embedding_table.parallel_optimizer = False

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)
        self.token_type_embeddings.gather.shard(((1, 1), (parallel_config.dp,)))
        self.token_type_embeddings.expand.shard(((parallel_config.dp, 1),))
        self.token_type_embeddings.embedding_table.parallel_optimizer = False
        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.layernorm = LayerNorm((config.hidden_size,), parallel_config.dp)
        self.dropout = Dropout(1 - config.hidden_dropout_prob).shard(((parallel_config.dp, 1, 1),))
        self.zeros_like = P.ZerosLike().shard(((parallel_config.dp, 1),))
        self.add = P.TensorAdd().shard(((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.add1 = P.TensorAdd().shard(((parallel_config.dp, 1, 1), (1, 1, 1)))
        self.full_batch = config.full_batch
        self.stride_slice_1 = P.StridedSlice().shard(((1, 1, 1),))

    def construct(self, input_ids, position_ids, token_type_ids=None):
        """ construct """

        if token_type_ids is None:
            token_type_ids = self.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if not self.full_batch:
            position_embeddings = self.stride_slice_1(position_embeddings, (0, 0, 0),
                                                      (1, position_embeddings.shape[1], position_embeddings.shape[2]),
                                                      (1, 1, 1))

        embeddings = self.add1(words_embeddings, position_embeddings)
        embeddings = self.add(embeddings, token_type_embeddings)
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class UniterImageEmbeddings(nn.Cell):
    """ UniterImageEmbeddings """

    def __init__(self, config, img_dim, parallel_config, is_parallel):
        super().__init__()
        self.img_linear = nn.Dense(img_dim, config.hidden_size).to_float(mstype.float16)
        self.img_linear.matmul.shard(((parallel_config.dp, 1), (1, 1)))
        self.img_linear.bias_add.shard(((parallel_config.dp, 1), (1,)))
        self.img_linear.weight.parallel_optimizer = False
        self.img_linear.bias.parallel_optimizer = False

        self.img_layer_norm = LayerNorm((config.hidden_size,), parallel_config.dp).to_float(mstype.float32)

        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)
        self.mask_embedding.gather.shard(((1, 1), (parallel_config.dp,)))
        self.mask_embedding.expand.shard(((parallel_config.dp, 1),))
        self.mask_embedding.embedding_table.parallel_optimizer = False
        self.add = P.TensorAdd().shard(((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.add1 = P.TensorAdd().shard(((parallel_config.dp, 1, 1), (1, 1, 1)))

        # tf naming convention for layer norm
        self.LayerNorm = LayerNorm((config.hidden_size,), parallel_config.dp)
        self.dropout = Dropout(1 - config.hidden_dropout_prob).shard(((parallel_config.dp, 1, 1),))
        self.cast = P.Cast()

        self.use_vit = getattr(config, 'use_vit', False)
        if self.use_vit:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.position_embeddings.gather.shard(((1, 1), (1,)))
            self.position_embeddings.expand.shard(((1, 1),))
            self.position_embeddings.embedding_table.parallel_optimizer = False

            self.full_batch = config.full_batch
            self.stride_slice_1 = P.StridedSlice().shard(((1, 1, 1),))
            
            self.vit_type = getattr(config, 'vit_type', 0)
            if self.vit_type == 0:
                vit_model = VitEval(batch_size=config.batch_size,
                                    patch_size=config.patch_size, image_size=config.train_image_size)
            elif self.vit_type == 1:
                vit_model = VitEvalV2(batch_size=config.batch_size,
                                    patch_size=config.patch_size, image_size=config.train_image_size)
            else:
                raise Exception("Error Vit Type")
            self.vit = vit_model

        else:
            self.pos_layer_norm = LayerNorm((config.hidden_size,), parallel_config.dp).to_float(mstype.float32)
            self.pos_linear = nn.Dense(7, config.hidden_size).to_float(mstype.float16)
            self.pos_linear.matmul.shard(((parallel_config.dp, 1), (1, 1)))
            self.pos_linear.bias_add.shard(((parallel_config.dp, 1), (1,)))
            self.pos_linear.weight.parallel_optimizer = False
            self.pos_linear.bias.parallel_optimizer = False

    def construct(self, img_feat, img_pos_feat, type_embeddings, img_masks=None, images=None):
        """ construct """

        
        if self.use_vit:
            if self.vit_type == 0:
                img_feat = self.vit(img_feat)
            elif self.vit_type == 1:
                img_feat = self.vit(images)

        ori_img_feat = img_feat

        if img_masks is not None:
            mask = self.mask_embedding(self.cast(img_masks, mstype.int32))
            img_feat = self.add(img_feat, mask)

        transformed_im = self.img_layer_norm(self.img_linear(img_feat))

        if self.use_vit:
            transformed_pos = self.position_embeddings(img_pos_feat)
            if not self.full_batch:
                transformed_pos = self.stride_slice_1(transformed_pos, (0, 0, 0),
                                                      (1, transformed_pos.shape[1], transformed_pos.shape[2]),
                                                      (1, 1, 1))
            embeddings = self.add1(transformed_im, transformed_pos)
        else:
            transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat))
            embeddings = self.add(transformed_im, transformed_pos)

        embeddings = self.add(embeddings, type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, ori_img_feat

class UniterAudioEmbeddings(nn.Cell):
    """ UniterAudioEmbeddings """

    def __init__(self, config, audio_dim, parallel_config):
        super().__init__()
        self.audio_linear = nn.Dense(audio_dim, config.hidden_size).to_float(mstype.float16)
        self.audio_linear.matmul.shard(((parallel_config.dp, 1), (1, 1)))
        self.audio_linear.bias_add.shard(((parallel_config.dp, 1), (1,)))
        self.audio_linear.weight.parallel_optimizer = False
        self.audio_linear.bias.parallel_optimizer = False

        self.audio_layer_norm = LayerNorm((config.hidden_size,), parallel_config.dp)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.position_embeddings.gather.shard(((1, 1), (1,)))
        self.position_embeddings.expand.shard(((1, 1),))
        self.position_embeddings.embedding_table.parallel_optimizer = False

        self.mask_embedding = nn.Embedding(2, audio_dim, padding_idx=0)
        self.mask_embedding.gather.shard(((1, 1), (parallel_config.dp,)))
        self.mask_embedding.expand.shard(((parallel_config.dp, 1),))
        self.mask_embedding.embedding_table.parallel_optimizer = False

        # tf naming convention for layer norm
        self.LayerNorm = LayerNorm((config.hidden_size,), parallel_config.dp)
        self.dropout = Dropout(1 - config.hidden_dropout_prob).shard(((parallel_config.dp, 1, 1),))
        self.add = P.TensorAdd().shard(((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.add1 = P.TensorAdd().shard(((1, 1, 1), (parallel_config.dp, 1, 1)))
        self.cast = P.Cast()
        self.full_batch = config.full_batch
        self.stride_slice_1 = P.StridedSlice().shard(((1, 1, 1),))

    def construct(self, audio_feat, audio_pos_ids, type_embeddings, audio_masks=None):
        """ construct """

        if audio_masks is not None:
            mask = self.mask_embedding(self.cast(audio_masks, mstype.int32))
            audio_feat = self.add(audio_feat, mask)

        transformed_audio = self.audio_layer_norm(self.audio_linear(audio_feat))
        position_embeddings = self.position_embeddings(audio_pos_ids)
        if not self.full_batch:
            position_embeddings = self.stride_slice_1(position_embeddings, (0, 0, 0),
                                                      (1, position_embeddings.shape[1], position_embeddings.shape[2]),
                                                      (1, 1, 1))
        embeddings = self.add(transformed_audio, self.add1(position_embeddings, type_embeddings))
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterThreeModel(nn.Cell):
    """ Modification for Joint Vision-Language Encoding
    """

    def __init__(self, config, img_dim, audio_dim, use_video=False, parallel_config=None, use_moe=False,
                 is_parallel=True):
        super().__init__(config)

        self.embeddings = UniterTextEmbeddings(config, parallel_config)

        self.img_embeddings = UniterImageEmbeddings(config, img_dim, parallel_config, is_parallel=is_parallel)
        self.audio_embeddings = UniterAudioEmbeddings(config, audio_dim, parallel_config)
        self.encoder = UniterEncoder(config, parallel_config, use_moe, is_parallel=is_parallel)
        self.pooler = BertPooler(config, parallel_config)

        self.gather = P.GatherD().shard(((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.cat = P.Concat(axis=1).shard(((parallel_config.dp, 1, 1),
                                           (parallel_config.dp, 1, 1),
                                           (parallel_config.dp, 1, 1)))
        self.cat1 = P.Concat(axis=1).shard(((parallel_config.dp, 1, 1),
                                            (parallel_config.dp, 1, 1)
                                            ))
        self.cast = P.Cast()
        self.ones_like = P.OnesLike().shard(((parallel_config.dp, 1),))
        self.mul = P.Mul().shard(((parallel_config.dp, 1), ()))
        self.unsqueeze = P.ExpandDims().shard(((parallel_config.dp, 1),))
        self.broadcastto = P.BroadcastTo((-1, C.MAX_FULL_LEN, config.hidden_size)).shard(
            ((parallel_config.dp, 1, 1),))
        self.broadcastto1 = P.BroadcastTo((-1, C.MAX_IMG_TEXT_LEN, config.hidden_size)).shard(
            ((parallel_config.dp, 1, 1),))
        self.hidden_size = config.hidden_size

        self.stride_slice = P.StridedSlice().shard(((parallel_config.dp, 1, 1),))
        self.squeeze = P.Squeeze(2).shard(((parallel_config.dp, 1, 1),))

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None):
        """ _compute_txt_embeddings """

        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat, img_masks=None,
                                img_type_ids=None, images=None):
        """ _compute_img_embeddings """

        if img_type_ids is None:
            img_type_ids = self.ones_like(self.cast(self.squeeze(self.stride_slice(img_feat, (0, 0, 0),
                                                                                   (img_feat.shape[0],
                                                                                    img_feat.shape[1], 1),
                                                                                   (1, 1, 1))), mstype.int32))

        img_type_embeddings = self.embeddings.token_type_embeddings(self.cast(img_type_ids, mstype.int32))
        output = self.img_embeddings(img_feat, img_pos_feat,
                                     img_type_embeddings, img_masks, images)
        return output

    def _compute_audio_embeddings(self, audio_feat=None, audio_pos_ids=None,
                                  audio_masks=None, audio_type_ids=None):
        """ _compute_audio_embeddings """

        if audio_type_ids is None:
            # audio_type_ids = self.fill(mstype.int32, audio_feat[:,:,0].shape, 2)
            audio_type_ids = self.mul(self.ones_like(self.squeeze(self.stride_slice(audio_feat, (0, 0, 0),
                                                                                    (audio_feat.shape[0],
                                                                                     audio_feat.shape[1], 1),
                                                                                    (1, 1, 1)))), 2)
        audio_type_embeddings = self.embeddings.token_type_embeddings(self.cast(audio_type_ids, mstype.int32))
        output = self.audio_embeddings(audio_feat, audio_pos_ids, audio_type_embeddings, audio_masks)
        return output

    def _compute_img_txt_embeddings(self, input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    gather_index, img_masks=None,
                                    txt_type_ids=None, img_type_ids=None,
                                    audio_feat=None, audio_pos_ids=None,
                                    audio_masks=None, audio_type_ids=None, images=None):
        """ _compute_img_txt_audio_embeddings """

        txt_emb = self._compute_txt_embeddings(
            input_ids, position_ids, txt_type_ids)

        img_emb, ori_img_feat = self._compute_img_embeddings(
            img_feat, img_pos_feat, img_masks, img_type_ids, images)

        gather_index = self.unsqueeze(gather_index, -1)
        gather_index = P.Cast()(self.broadcastto1(gather_index), mstype.int32)

        # batch_size * (txt_len + img_len) * hidden_size
        # 32, 236, 768 -> 32, 191, 768
        concat_tmp = self.cat1((txt_emb, img_emb))
        embedding_output = self.gather(concat_tmp, 1, gather_index)

        return embedding_output, ori_img_feat

    def _compute_img_txt_audio_embeddings(self, input_ids, position_ids,
                                          img_feat, img_pos_feat,
                                          gather_index, img_masks=None,
                                          txt_type_ids=None, img_type_ids=None,
                                          audio_feat=None, audio_pos_ids=None,
                                          audio_masks=None, audio_type_ids=None, images=None):
        """ _compute_img_txt_audio_embeddings """

        txt_emb = self._compute_txt_embeddings(
            input_ids, position_ids, txt_type_ids)

        img_emb, ori_img_feat = self._compute_img_embeddings(
            img_feat, img_pos_feat, img_masks, img_type_ids, images)

        audio_emb = self._compute_audio_embeddings(
            audio_feat, audio_pos_ids, audio_masks, audio_type_ids)

        gather_index = self.unsqueeze(gather_index, -1)
        gather_index = P.Cast()(self.broadcastto(gather_index), mstype.int32)

        # batch_size * (txt_len + img_len + audio_len) * hidden_size
        # 32, 236, 768 -> 32, 191, 768
        concat_tmp = self.cat((txt_emb, img_emb, audio_emb))
        embedding_output = self.gather(concat_tmp, 1, gather_index)

        return embedding_output, ori_img_feat

    def construct(self, input_ids, position_ids,
                  img_feat, img_pos_feat,
                  attention_mask, gather_index=None, img_masks=None,
                  output_all_encoded_layers=True,
                  txt_type_ids=None, img_type_ids=None,
                  audio_feat=None, audio_pos_ids=None,
                  audio_type_ids=None, audio_masks=None, images=None
                  ):
        """
        construct
        """

        if input_ids is None and audio_feat is None:
            embedding_output, _ = self._compute_img_embeddings(
                img_feat, img_pos_feat, img_masks, img_type_ids)
            ori_img_feat = None
        elif input_ids is not None and img_feat is None and audio_feat is None:
            embedding_output = self._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids)
            ori_img_feat = None
        elif input_ids is not None and audio_feat is None:
            embedding_output, ori_img_feat = self._compute_img_txt_embeddings(
               input_ids, position_ids, img_feat, img_pos_feat, gather_index,
                img_masks, txt_type_ids, img_type_ids, images)
        else:
            embedding_output, ori_img_feat = self._compute_img_txt_audio_embeddings(
                input_ids, position_ids,
                img_feat, img_pos_feat,
                gather_index, img_masks, txt_type_ids, img_type_ids,
                audio_feat, audio_pos_ids, audio_masks, audio_type_ids, images)

        embedding_output = self.cast(embedding_output, mstype.float16)
        attention_mask = self.cast(attention_mask, mstype.float16)
        encoded_layers, moe_loss = self.encoder(embedding_output, attention_mask)

        return encoded_layers, moe_loss, ori_img_feat


class UniterThreeModelAudio(nn.Cell):
    """ Modification for Joint Vision-Language Encoding
    """

    def __init__(self, config, parallel_config=None, use_moe=False, is_parallel=True):
        super().__init__(config)

        self.embeddings = UniterTextEmbeddings(config, parallel_config)

        self.encoder = UniterEncoder(config, parallel_config, use_moe, is_parallel)
        self.pooler = BertPooler(config, parallel_config)

        self.gather = P.GatherD().shard(((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.cat = P.Concat(axis=1).shard(((parallel_config.dp, 1, 1),
                                           (parallel_config.dp, 1, 1),
                                           (parallel_config.dp, 1, 1)))
        self.cast = P.Cast()
        self.ones_like = P.OnesLike().shard(((parallel_config.dp, 1),))
        self.mul = P.Mul().shard(((parallel_config.dp, 1), ()))
        self.unsqueeze = P.ExpandDims().shard(((parallel_config.dp, 1),))
        self.broadcastto = P.BroadcastTo((-1, 30, config.hidden_size)).shard(((parallel_config.dp, 1, 1),))
        self.hidden_size = config.hidden_size

        self.stride_slice = P.StridedSlice().shard(((parallel_config.dp, 1, 1),))
        self.squeeze = P.Squeeze(2).shard(((parallel_config.dp, 1, 1),))

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None):
        """ _compute_txt_embeddings """

        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def construct(self, input_ids, position_ids, attention_mask):
        """ construct """

        embedding_output = self._compute_txt_embeddings(input_ids, position_ids)

        embedding_output = self.cast(embedding_output, mstype.float16)
        attention_mask = self.cast(attention_mask, mstype.float16)
        encoded_layers, moe_loss = self.encoder(embedding_output, attention_mask)

        return encoded_layers, moe_loss

