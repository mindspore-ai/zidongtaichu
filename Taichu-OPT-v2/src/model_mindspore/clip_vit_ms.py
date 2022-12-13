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

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
import numpy as np
import pickle

class LayerNorm(nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def construct(self, x: ms.Tensor):
        y = super().construct(P.Cast()(x, ms.float32))
        y = P.Cast()(y, x.dtype)
        return y

class Linear(nn.Dense):

    def construct(self, x: ms.Tensor):
        y = super().construct(P.Cast()(x, ms.float16))
        y = P.Cast()(y, ms.float32)
        return y


class CLIPAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size, num_attention_heads, attention_dropout):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** -0.5
        self.dropout_prob = attention_dropout

        self.k_proj = Linear(self.embed_dim, self.embed_dim).to_float(ms.float16)
        self.v_proj = Linear(self.embed_dim, self.embed_dim).to_float(ms.float16)
        self.q_proj = Linear(self.embed_dim, self.embed_dim).to_float(ms.float16)
        self.out_proj = Linear(self.embed_dim, self.embed_dim).to_float(ms.float16)

        self.transpose = P.Transpose()
        self.bmm = P.BatchMatMul(transpose_a=False, transpose_b=True)
        self.bmm1 = P.BatchMatMul(transpose_a=False, transpose_b=False)
        self.softmax = nn.Softmax(axis=-1)
        self.dropout = nn.Dropout(keep_prob=(1 - self.dropout_prob))

    def _shape(self, tensor, seq_len, bsz):
        value = tensor.view((bsz, seq_len, self.num_heads, self.head_dim))
        value = self.transpose(value, (0, 2, 1, 3))
        return value

    def construct(
            self,
            hidden_states,
            attention_mask=None,
            causal_attention_mask=None):
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(proj_shape)
        key_states = key_states.view(proj_shape)
        value_states = value_states.view(proj_shape)

        src_len = key_states.shape[1]

        attn_weights = self.bmm(query_states.astype(ms.float16), key_states.astype(ms.float16)).astype(ms.float32)

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view((bsz, self.num_heads, tgt_len, src_len)) + causal_attention_mask
            attn_weights = attn_weights.view((bsz * self.num_heads, tgt_len, src_len))

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view((bsz, self.num_heads, tgt_len, src_len)) + attention_mask
            attn_weights = attn_weights.view((bsz * self.num_heads, tgt_len, src_len))

        attn_weights = self.softmax(attn_weights)

        attn_probs = self.dropout(attn_weights)

        attn_output = self.bmm1(attn_probs.astype(ms.float16), value_states.astype(ms.float16)).astype(ms.float32)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view((bsz, self.num_heads, tgt_len, self.head_dim))
        attn_output = self.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.view((bsz, tgt_len, embed_dim))

        attn_output = self.out_proj(attn_output).astype(ms.float32)

        return attn_output

class QuickGELUActivation(nn.Cell):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = P.Sigmoid()

    def construct(self, input):
        return input * self.sigmoid(1.702 * input)

class CLIPMLP(nn.Cell):
    def __init__(self, hidden_act, hidden_size, intermediate_size):
        super().__init__()
        # self.activation_fn = P.FastGeLU()
        self.activation_fn = QuickGELUActivation()
        self.fc1 = Linear(hidden_size, intermediate_size).to_float(ms.float16)
        self.fc2 = Linear(intermediate_size, hidden_size).to_float(ms.float16)

    def construct(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Cell):
    def __init__(self, hidden_size, hidden_act, num_attention_heads, attention_dropout, intermediate_size):
        super().__init__()
        self.self_attn = CLIPAttention(hidden_size, num_attention_heads, attention_dropout)
        self.layer_norm1 = LayerNorm([hidden_size], epsilon=1e-05)
        self.mlp = CLIPMLP(hidden_act, hidden_size, intermediate_size)
        self.layer_norm2 = LayerNorm([hidden_size], epsilon=1e-05)

    def construct(self, hidden_states, attention_mask):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=None
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Cell):

    def __init__(self, hidden_size, hidden_act, num_attention_heads, attention_dropout, intermediate_size,
                 num_hidden_layers):
        super().__init__()
        self.depth = num_hidden_layers
        self.layers = nn.CellList(
            [CLIPEncoderLayer(hidden_size, hidden_act, num_attention_heads, attention_dropout, intermediate_size) for _
             in range(num_hidden_layers)])

    def construct(self, inputs_embeds):
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states, attention_mask=None)

        return hidden_states


class CLIPVisionTransformer(nn.Cell):
    def __init__(self, image_size, patch_size, hidden_size, hidden_act, num_attention_heads, attention_dropout,
                 intermediate_size, num_hidden_layers):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.num_patch_embed = (self.image_size // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=hidden_size, kernel_size=self.patch_size,
            stride=self.patch_size, has_bias=False
        )
        self.class_embedding = Parameter(np.random.randn(hidden_size).astype(np.float32))
        self.num_pos_embed = self.num_patch_embed + 1
        self.pos_embed = nn.Embedding(self.num_pos_embed, hidden_size)
        self.position_ids = Parameter(np.expand_dims(np.arange(self.num_pos_embed).astype(np.int32), 0), requires_grad=False)

        self.pre_layrnorm = LayerNorm([hidden_size], epsilon=1e-5)
        self.encoder = CLIPEncoder(hidden_size, hidden_act, num_attention_heads, attention_dropout, intermediate_size,
                                   num_hidden_layers)
        self.post_layernorm = LayerNorm([hidden_size], epsilon=1e-5)

        self.tranpose = P.Transpose()
        self.cat = P.Concat(axis=1)
        self.reshape = P.Reshape()

    def construct(self, x):
        batch_size = x.shape[0]
        patch_embeds = self.patch_embed(x)  # shape = [*, width, grid, grid]
        patch_embeds = self.reshape(patch_embeds, (patch_embeds.shape[0], patch_embeds.shape[1], patch_embeds.shape[2] * patch_embeds.shape[3]))
        patch_embeds = self.tranpose(patch_embeds, (0, 2, 1))

        class_embeds = P.BroadcastTo((batch_size, 1, -1))(self.class_embedding)
        embeddings = self.cat([class_embeds, patch_embeds])
        hidden_states = embeddings + self.pos_embed(self.position_ids)

        hidden_states = self.pre_layrnorm(hidden_states)
        outputs = self.encoder(inputs_embeds=hidden_states)
        outputs = self.post_layernorm(outputs)

        return outputs


def update_param(net_param_dict, params, ms_full_name, torch_full_name):
    old_param = net_param_dict[ms_full_name]
    # print(ms_full_name, old_param.data.dtype, params[torch_full_name].dtype)
    new_param = ms.Tensor(params[torch_full_name], old_param.data.dtype)
    old_param.set_data(new_param)


def load_visual_encoder(net, param_dict, layers_num=12):

    update_count = 0

    for mindspore_full_name, torch_full_name in [
        ('class_embedding', 'vision_model.embeddings.class_embedding'),
        ('position_ids', 'vision_model.embeddings.position_ids'),
        ('patch_embed.weight', 'vision_model.embeddings.patch_embedding.weight'),
        ('pos_embed.embedding_table', 'vision_model.embeddings.position_embedding.weight'),
        ('pre_layrnorm.gamma', 'vision_model.pre_layrnorm.weight'),
        ('pre_layrnorm.beta', 'vision_model.pre_layrnorm.bias'),
        ('post_layernorm.gamma', 'vision_model.post_layernorm.weight'),
        ('post_layernorm.beta', 'vision_model.post_layernorm.bias'),
    ]:
        update_param(net, param_dict, mindspore_full_name, torch_full_name)
        update_count += 1

    for i in range(layers_num):
        mindspore_prefix = 'encoder.layers.'
        torch_prefix = 'vision_model.encoder.layers.'
        for mindspore_name, torch_name in [
            ('self_attn.k_proj.weight', 'self_attn.k_proj.weight'),
            ('self_attn.k_proj.bias', 'self_attn.k_proj.bias'),
            ('self_attn.v_proj.weight', 'self_attn.v_proj.weight'),
            ('self_attn.v_proj.bias', 'self_attn.v_proj.bias'),
            ('self_attn.q_proj.weight', 'self_attn.q_proj.weight'),
            ('self_attn.q_proj.bias', 'self_attn.q_proj.bias'),
            ('self_attn.out_proj.weight', 'self_attn.out_proj.weight'),
            ('self_attn.out_proj.bias', 'self_attn.out_proj.bias'),
            ('layer_norm1.gamma', 'layer_norm1.weight'),
            ('layer_norm1.beta', 'layer_norm1.bias'),
            ('mlp.fc1.weight', 'mlp.fc1.weight'),
            ('mlp.fc1.bias', 'mlp.fc1.bias'),
            ('mlp.fc2.weight', 'mlp.fc2.weight'),
            ('mlp.fc2.bias', 'mlp.fc2.bias'),
            ('layer_norm2.gamma', 'layer_norm2.weight'),
            ('layer_norm2.beta', 'layer_norm2.bias'),
        ]:
            mindspore_full_name = '{}{}.{}'.format(mindspore_prefix, i, mindspore_name)
            torch_full_name = '{}{}.{}'.format(torch_prefix, i, torch_name)
            update_param(net, param_dict, mindspore_full_name, torch_full_name)
            update_count += 1

    print(f"update_count {update_count}")


def load_clip_vit(image_size=224, patch_size=32, hidden_size=768,
                  hidden_act="", num_attention_heads=12, attention_dropout=0.1,
                  intermediate_size=3072, num_hidden_layers=12, ckpt_path=""):

    visual_encoder = CLIPVisionTransformer(image_size, patch_size, hidden_size, hidden_act, num_attention_heads,
                                           attention_dropout, intermediate_size, num_hidden_layers)

    if ckpt_path is not None and len(ckpt_path) > 0:
        with open(ckpt_path, 'rb') as ckpt_fp:
            param_dict = pickle.load(ckpt_fp)

        visual_count = 0
        for key, val in param_dict.items():
            if key.startswith("vision_model"):
                visual_count += 1

        print(f"visual_count {visual_count}")

        load_visual_encoder(visual_encoder.parameters_dict(), param_dict, layers_num=num_hidden_layers)

    return visual_encoder


def load_clip_vit_base_patch32(ckpt_path):

    image_size = 224
    patch_size = 32
    hidden_size = 768
    hidden_act = ""
    num_attention_heads = 12
    attention_dropout = 0.1
    intermediate_size = 3072
    num_hidden_layers = 12

    visual_encoder = load_clip_vit(image_size=image_size,
                                   patch_size=patch_size,
                                   hidden_size=hidden_size,
                                   hidden_act=hidden_act,
                                   num_attention_heads=num_attention_heads,
                                   attention_dropout=attention_dropout,
                                   intermediate_size=intermediate_size,
                                   num_hidden_layers=num_hidden_layers,
                                   ckpt_path=ckpt_path)

    return visual_encoder

def load_clip_vit_base_patch16(ckpt_path):

    image_size = 224
    patch_size = 16
    hidden_size = 768
    hidden_act = ""
    num_attention_heads = 12
    attention_dropout = 0.1
    intermediate_size = 3072
    num_hidden_layers = 12

    visual_encoder = load_clip_vit(image_size=image_size,
                                   patch_size=patch_size,
                                   hidden_size=hidden_size,
                                   hidden_act=hidden_act,
                                   num_attention_heads=num_attention_heads,
                                   attention_dropout=attention_dropout,
                                   intermediate_size=intermediate_size,
                                   num_hidden_layers=num_hidden_layers,
                                   ckpt_path=ckpt_path)

    return visual_encoder

def load_clip_vit_large_patch14(ckpt_path):

    image_size = 224
    patch_size = 14
    hidden_size = 1024
    hidden_act = ""
    num_attention_heads = 16
    attention_dropout = 0.1
    intermediate_size = 4096
    num_hidden_layers = 24

    visual_encoder = load_clip_vit(image_size=image_size,
                                   patch_size=patch_size,
                                   hidden_size=hidden_size,
                                   hidden_act=hidden_act,
                                   num_attention_heads=num_attention_heads,
                                   attention_dropout=attention_dropout,
                                   intermediate_size=intermediate_size,
                                   num_hidden_layers=num_hidden_layers,
                                   ckpt_path=ckpt_path)

    return visual_encoder


def load_clip_vit_large_patch14_336(ckpt_path):

    image_size = 336
    patch_size = 14
    hidden_size = 1024
    hidden_act = ""
    num_attention_heads = 16
    attention_dropout = 0.1
    intermediate_size = 4096
    num_hidden_layers = 24

    visual_encoder = load_clip_vit(image_size=image_size,
                                   patch_size=patch_size,
                                   hidden_size=hidden_size,
                                   hidden_act=hidden_act,
                                   num_attention_heads=num_attention_heads,
                                   attention_dropout=attention_dropout,
                                   intermediate_size=intermediate_size,
                                   num_hidden_layers=num_hidden_layers,
                                   ckpt_path=ckpt_path)

    return visual_encoder

