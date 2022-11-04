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

from easydict import EasyDict as edict
import mindspore.common.dtype as mstype
from . import config

cfg = edict({
    'transformer_network': 'large',
    'init_loss_scale_value': 1024,
    'scale_factor': 2,
    'scale_window': 2000,
    'optimizer': 'Adam',
    'optimizer_adam_beta2': 0.997,
    'lr_schedule': edict({
        'learning_rate': 2.0,
        'warmup_steps': 8000,
        'start_decay_step': 16000,
        'min_lr': 0.0,
    }),
})


class TransformerConfig:
    """
    Configuration for `Transformer`.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input sequence. Default: 128.
        vocab_size (int): The shape of each embedding vector. Default: 36560.
        hidden_size (int): Size of the layers. Default: 1024.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder/decoder
                           cell. Default: 6.
        num_attention_heads (int): Number of attention heads in the Transformer
                             encoder/decoder cell. Default: 16.
        intermediate_size (int): Size of intermediate layer in the Transformer
                           encoder/decoder cell. Default: 4096.
        hidden_act (str): Activation function used in the Transformer encoder/decoder
                    cell. Default: "relu".
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.3.
        attention_probs_dropout_prob (float): The dropout probability for
                                      MultiheadAttention. Default: 0.3.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 128.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        label_smoothing (float): label smoothing setting. Default: 0.1
        beam_width (int): beam width setting. Default: 4
        max_decode_length (int): max decode length in evaluation. Default: 80
        length_penalty_weight (float): normalize scores of translations according to their length. Default: 1.0
        dtype (:class:`mindspore.dtype`): Data type of the input. Default: mstype.float32.
        compute_type (:class:`mindspore.dtype`): Compute type in Transformer. Default: mstype.float32.
    """

    def __init__(self,
                 batch_size=100,
                 seq_length=150,
                 vocab_size=36560,
                 hidden_size=1024,
                 num_hidden_layers=6,
                 num_attention_heads=16,
                 intermediate_size=4096,
                 hidden_act="relu",
                 hidden_dropout_prob=0.3,
                 attention_probs_dropout_prob=0.3,
                 max_position_embeddings=150,
                 initializer_range=0.02,
                 label_smoothing=0.1,
                 beam_width=1,
                 max_decode_length=config.MAX_FULL_TEXT_LEN,
                 length_penalty_weight=1.0,
                 dtype=mstype.float32,
                 compute_type=mstype.float32):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.label_smoothing = label_smoothing
        self.beam_width = beam_width
        self.max_decode_length = max_decode_length
        self.length_penalty_weight = length_penalty_weight
        self.dtype = dtype
        self.compute_type = compute_type
'''
two kinds of transformer model version
'''
if cfg.transformer_network == 'large':
    transformer_net_cfg = TransformerConfig(
        batch_size=4,
        seq_length=config.MAX_FULL_LEN,
        vocab_size=36560,
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="relu",
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        max_position_embeddings=150,
        initializer_range=0.02,
        label_smoothing=0.1,
        dtype=mstype.float32,
        compute_type=mstype.float16,
        beam_width=1)
    transformer_net_cfg_gpu = TransformerConfig(
        seq_length=config.MAX_FULL_LEN,
        vocab_size=36560,
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="relu",
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        max_position_embeddings=150,
        initializer_range=0.02,
        label_smoothing=0.1,
        dtype=mstype.float32,
        compute_type=mstype.float16)
if cfg.transformer_network == 'base':
    transformer_net_cfg = TransformerConfig(
        batch_size=96,
        seq_length=config.MAX_FULL_LEN,
        vocab_size=36560,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="relu",
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2,
        max_position_embeddings=150,
        initializer_range=0.02,
        label_smoothing=0.1,
        dtype=mstype.float32,
        compute_type=mstype.float16)



transformer_net_cfg_base = TransformerConfig(
    batch_size=96,
    seq_length=config.MAX_FULL_LEN,
    vocab_size=36560,
    hidden_size=512,
    num_hidden_layers=4,
    num_attention_heads=8,
    intermediate_size=2048,
    hidden_act="relu",
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2,
    max_position_embeddings=150,
    initializer_range=0.02,
    label_smoothing=0.1,
    dtype=mstype.float32,
    compute_type=mstype.float16,
    beam_width=1)
