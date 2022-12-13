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
""" model_ms """

import logging
import numpy as np
from src.config import config as C
import mindspore
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.parallel.nn import TransformerEncoder, TransformerDecoder
from mindspore.nn.transformer import TransformerOpParallelConfig
from mindspore.communication.management import get_group_size
from mindspore.nn import Cell
from src.model_mindspore.parallel_transformer import Dropout, LayerNorm, ParallelConfig
from src.model_mindspore.clip_vit_ms import load_clip_vit_base_patch16, load_clip_vit_large_patch14_336

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

class MemoryAttentionMask(Cell):
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
        super(MemoryAttentionMask, self).__init__()
        self.reshape = P.Reshape()
        self.mul = P.BatchMatMul().shard(
            ((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.expand_dim = P.ExpandDims().shard(((1, 1),))
        self.right_mask = mindspore.Tensor(np.ones(C.MAX_IMG_LEN), mindspore.float16)

    def construct(self, input_mask):
        r"""
        Generate the attention mask matrix.
        """

        shape_left = P.Shape()(input_mask) + (1,)

        right_mask = P.BroadcastTo((shape_left[0], -1))(self.right_mask)
        right_shape = P.Shape()(right_mask)
        shape_right = (right_shape[0], 1, right_shape[1])

        # Mask the padded inputs
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(right_mask, shape_right)
        attention_mask = self.mul(mask_left.astype(mstype.float16), mask_right).astype(mstype.float32)

        # [bs, tgt_length, src_length]
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


def _get_lambda_func(total_layer=None):
    r"""
        A wrapper function of specifying pipeline stage and gradient aggregation fusion. If the total layer
        is not None, for example, set in the transformer model, the pipeline stage setting function will be
        `(layer_id + 0) // (total_layers / parallel_config.pipeline_stage)` for the encoder and,
        `(layer_id + offset) //
        (total_layers / parallel_config.pipeline_stage)` for the decoder, where `offset` is the layers in the encoder.
    """

    def _set_parallel_configure_for_layer(network, layer_id, offset, parallel_config, layers):
        r"""
            Default setting for the pipeline is: `(layer_id + offset) // (layers / pipeline_stage)`.

            Args:
                network(Cell) - Represents the transformer block
                layer_id(int) - Means the layer index for the current module, counts from zero.
                offset(int) - Means the layer_index needs an offset, if there are other modules in the net.
                layers(int) - The total layers used for the model.
        """

        start = 3

        if layer_id < start:
            network.pipeline_stage = 0
        elif layer_id < layers - start*2:
            network.pipeline_stage = (layer_id-start*2)//6+start
        else:
            network.pipeline_stage = 7

        # Used for optimizer's fusion tag
        dis = max(int(layers / parallel_config.gradient_aggregation_group), 1)
        network.set_comm_fusion(int((layer_id + offset) / dis) + 1)
        # Used for enabling recomputation of the block
        if isinstance(parallel_config.recompute, bool):
            if parallel_config.recompute:
                network.recompute()
        else:
            if parallel_config.recompute.recompute:
                paralel_op_comm_compute = parallel_config.recompute.parallel_optimizer_comm_recompute
                network.recompute(parallel_optimizer_comm_recompute=paralel_op_comm_compute,
                                  mp_comm_recompute=parallel_config.recompute.mp_comm_recompute,
                                  recompute_slice_activation=parallel_config.recompute.recompute_slice_activation)

    return _set_parallel_configure_for_layer


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
        if is_parallel:
            self.group_size = get_group_size()
        else:
            self.group_size = 1

        if config.use_pipeline:
            lambda_func = _get_lambda_func()
        else:
            lambda_func = None

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
                                          parallel_config=op_parallel_config,
                                          lambda_func=lambda_func)

    def construct(self, hidden_states, attention_mask):
        """ construct """

        attention_mask = self.attention_mask(attention_mask)

        hidden_states, past = self.encoder(hidden_states, attention_mask)

        return hidden_states, past


class UniterDecoder(nn.Cell):
    """ UniterEncoder """

    def __init__(self, config, parallel_config, use_moe, is_parallel):
        super().__init__()
        self.num_layers = config.num_hidden_layers

        self.attention_mask = BertAttentionMask(parallel_config)
        self.memory_mask = MemoryAttentionMask(parallel_config)

        op_parallel_config = TransformerOpParallelConfig(data_parallel=parallel_config.dp,
                                                         model_parallel=parallel_config.mp,
                                                         pipeline_stage=parallel_config.pipeline_stage,
                                                         optimizer_shard=parallel_config.optimizer_shard,
                                                         vocab_emb_dp=parallel_config.vocab_emb_dp)
        if is_parallel:
            self.group_size = get_group_size()
        else:
            self.group_size = 1

        if config.use_pipeline:
            lambda_func = _get_lambda_func()
        else:
            lambda_func = None

        self.decoder = TransformerDecoder(batch_size=config.batch_size,
                                          num_layers=config.num_hidden_layers,
                                          hidden_size=config.hidden_size,
                                          ffn_hidden_size=config.intermediate_size,
                                          num_heads=config.num_attention_heads,
                                          src_seq_length=C.MAX_IMG_LEN,
                                          tgt_seq_length=C.MAX_FULL_TEXT_LEN,
                                          attention_dropout_rate=config.attention_probs_dropout_prob,
                                          hidden_dropout_rate=config.hidden_dropout_prob,
                                          hidden_act=config.hidden_act,
                                          post_layernorm_residual=False,
                                          parallel_config=op_parallel_config,
                                          lambda_func=lambda_func)

    def construct(self, txt_feat, img_feat, att_txt):
        """ construct """

        # batch_size, seq_length, seq_length
        att_txt_input = self.attention_mask(att_txt)

        # batch_size, tgt_length, src_length
        att_mem_input = self.memory_mask(att_txt)

        # hidden_states, attention_mask, encoder_output = None, memory_mask = None,
        # init_reset = True, batch_valid_length = None
        hidden_states, past = self.decoder(txt_feat, att_txt_input, img_feat, att_mem_input)

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
        self.cast = P.Cast()

        if config.use_text_embed:
            if config.use_pipeline:
                lambda_func = _get_lambda_func()
            else:
                lambda_func = None

            op_parallel_config = TransformerOpParallelConfig(data_parallel=parallel_config.dp,
                                                             model_parallel=parallel_config.mp,
                                                             pipeline_stage=parallel_config.pipeline_stage,
                                                             optimizer_shard=parallel_config.optimizer_shard,
                                                             vocab_emb_dp=parallel_config.vocab_emb_dp)

            self.attention_mask = BertAttentionMask(parallel_config)
            self.encoder = TransformerEncoder(batch_size=config.batch_size,
                                              num_layers=config.num_hidden_layers,
                                              hidden_size=config.hidden_size,
                                              ffn_hidden_size=config.intermediate_size,
                                              num_heads=config.num_attention_heads,
                                              seq_length=C.MAX_FULL_TEXT_LEN,
                                              attention_dropout_rate=config.attention_probs_dropout_prob,
                                              hidden_dropout_rate=config.hidden_dropout_prob,
                                              hidden_act=config.hidden_act,
                                              post_layernorm_residual=False,
                                              parallel_config=op_parallel_config,
                                              lambda_func=lambda_func)
        else:
            self.encoder = None

    def construct(self, input_ids, position_ids, token_type_ids=None, attn_masks_text=None):
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

        if self.encoder is not None:

            embeddings = self.cast(embeddings, mstype.float16)
            attn_masks_text = self.cast(attn_masks_text, mstype.float16)

            attn_masks_text = self.attention_mask(attn_masks_text)
            embeddings, past = self.encoder(embeddings, attn_masks_text)

        return embeddings

def freeze_net(net):
    for param in net.get_parameters():
        param.requires_grad = False

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
        self.caption_task = getattr(config, 'caption_task', False)
        if self.use_vit:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.position_embeddings.gather.shard(((1, 1), (1,)))
            self.position_embeddings.expand.shard(((1, 1),))
            self.position_embeddings.embedding_table.parallel_optimizer = False

            self.full_batch = config.full_batch
            self.stride_slice_1 = P.StridedSlice().shard(((1, 1, 1),))

            # params_dict = load_checkpoint(config.vit_ckpt_file)
            # print("===============params_dict================", params_dict)
            if is_parallel:
                self.group_size = get_group_size()
            else:
                self.group_size = 1

            
            if not self.caption_task:
                self.vit_type = getattr(config, 'vit_type', "vit_base_patch16")
                self.vit_freeze = getattr(config, 'vit_freeze', False)

                print(f"load vit_type {self.vit_type} {config.vit_ckpt_file}")

                if self.vit_type == "vit_base_patch16":
                    vit_model = load_clip_vit_base_patch16(config.vit_ckpt_file)
                elif self.vit_type == "vit_base_patch14_336":
                    vit_model = load_clip_vit_large_patch14_336(config.vit_ckpt_file)
                else:
                    raise Exception("Error Vit Type")

                print("load vit_type success", config.vit_ckpt_file)

                # net_not_load = vit_model.init_weights(params_dict)
                # net_not_load = vit_model
                # print("===============net_not_load================", net_not_load)
                self.vit = vit_model
                if self.vit_freeze:
                    print("freeze vit")
                    freeze_net(self.vit)
                else:
                    print("not freeze vit")

        else:
            self.pos_layer_norm = LayerNorm((config.hidden_size,), parallel_config.dp).to_float(mstype.float32)
            self.pos_linear = nn.Dense(7, config.hidden_size).to_float(mstype.float16)
            self.pos_linear.matmul.shard(((parallel_config.dp, 1), (1, 1)))
            self.pos_linear.bias_add.shard(((parallel_config.dp, 1), (1,)))
            self.pos_linear.weight.parallel_optimizer = False
            self.pos_linear.bias.parallel_optimizer = False

    def get_feat(self, images):
        images = images.astype(mstype.float32)
        img_feat = self.vit(images)
        # img_feat = self.img_layer_norm(self.img_linear(img_feat))
        return img_feat

    def construct(self, img_feat, img_pos_feat, type_embeddings, img_masks=None, images=None):
        """ construct """

        if self.use_vit and not self.caption_task:
            img_feat = self.vit(images)

        ori_img_feat = img_feat

        if img_masks is not None:
            mask = self.mask_embedding(self.cast(img_masks, mstype.int32))
            # [10, 197, 768]
            # [10, 247, 768]
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

        if not self.caption_task:
            embeddings = self.add(embeddings, type_embeddings)
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings, ori_img_feat
        else:
            embeddings = self.LayerNorm(transformed_im)
            embeddings = self.dropout(embeddings)
            return embeddings


class UniterThreeModel(nn.Cell):
    """ Modification for Joint Vision-Language Encoding
    """

    def __init__(self, config, img_dim, audio_dim, parallel_config=None, use_moe=False,
                 is_parallel=True):
        super().__init__(config)

        self.embeddings = UniterTextEmbeddings(config, parallel_config)

        self.img_embeddings = UniterImageEmbeddings(config, img_dim, parallel_config, is_parallel=is_parallel)

        self.freeze_text = getattr(config, "freeze_text", False)
        if self.freeze_text:
            print("freeze text")
            freeze_net(self.embeddings)
        else:
            print("not freeze text")

        if config.use_pipeline:
            self.embeddings.pipeline_stage = 0
            self.img_embeddings.pipeline_stage = 0

        self.use_encoder_fusion = getattr(config, "use_encoder_fusion", True)
        print(f"use_encoder_fusion: {self.use_encoder_fusion}")
        if self.use_encoder_fusion:
            self.encoder = UniterEncoder(config, parallel_config, use_moe, is_parallel=is_parallel)
        else:
            self.encoder = UniterDecoder(config, parallel_config, use_moe, is_parallel=is_parallel)

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

        self.slice_first = P.StridedSlice().shard(((parallel_config.dp, 1, 1),))

    def get_first_token(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = self.slice_first(hidden_states, (0, 0, 0), (hidden_states.shape[0], 1, hidden_states.shape[2]),
                                        (1, 1, 1))
        first_token_tensor = first_token_tensor.view(hidden_states.shape[0], hidden_states.shape[2])
        return first_token_tensor

    def _compute_txt_embeddings(self, input_ids, position_ids, txt_type_ids=None, attn_masks_text=None):
        """ _compute_txt_embeddings """

        output = self.embeddings(input_ids, position_ids, txt_type_ids, attn_masks_text)

        return output

    def feat_fusion(self, txt_feat, img_feat, attention_mask, att_text):

        txt_feat = self.cast(txt_feat, mstype.float32)
        img_feat = self.cast(img_feat, mstype.float32)
        attention_mask = self.cast(attention_mask, mstype.float32)
        att_text = self.cast(att_text, mstype.float32)

        if self.use_encoder_fusion:
            embedding_output = self.cat1((img_feat, txt_feat))
            encoded_layers, moe_loss = self.encoder(embedding_output, attention_mask)
        else:
            encoded_layers, moe_loss = self.encoder(txt_feat, img_feat, att_text)

        return encoded_layers, moe_loss




