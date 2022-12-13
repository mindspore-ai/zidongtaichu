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
""" transformer_config """


from easydict import EasyDict as edict
import mindspore.common.dtype as mstype
from src.model_mindspore.transformer_model import TransformerConfig
from src.config import config

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
