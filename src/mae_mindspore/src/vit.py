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
from xml.etree.ElementPath import ops
import numpy as np

from mindspore import nn
from mindspore import Tensor
from mindspore import context
from mindspore import ops as P
from mindspore import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.parallel.nn import TransformerEncoder
from mindspore.train.serialization import load_param_into_net
from mindspore.common.initializer import initializer, XavierUniform
from mindspore.nn.transformer import TransformerOpParallelConfig
from src.model_mindspore.parallel_transformer import ParallelConfig, LayerNorm

from .modules import VitStem, PatchEmbed
from .mae_vit import MAEModule
from .layers import _LayerNorm, _Linear, _Dropout

class Vit(MAEModule):
    """
    pass
    """

    def __init__(self,
                 batch_size,
                 patch_size,
                 image_size,
                 encoder_layers=12,
                 encoder_num_heads=12,
                 encoder_dim=768,
                 mlp_ratio=4,
                 num_classes=1001,
                 dropout=0.1,
                 initialization=XavierUniform(),
                 return_feature=False):
        super(Vit, self).__init__(batch_size, image_size, patch_size)

        cls_token = Parameter(
            initializer(initialization, (1, 1, encoder_dim)),
            name='cls', requires_grad=True
        )
        self.cls_token = P.Tile()(cls_token, (batch_size, 1, 1))
        seq_length = self.num_patches + 1
        self.encoder_pos_embedding = Parameter(
            initializer(initialization, (1, seq_length, encoder_dim)),
            name='pos_embedding', requires_grad=True
        )

        self.encoder = TransformerEncoder(batch_size=batch_size, num_layers=encoder_layers,
                                          num_heads=encoder_num_heads, hidden_size=encoder_dim,
                                          ffn_hidden_size=encoder_dim * mlp_ratio, seq_length=seq_length)

        self.add = P.Add()
        self.cast = P.Cast()
        self.cat = P.Concat(axis=1)
        self.norm = nn.LayerNorm((encoder_dim,), epsilon=1e-5)
        self.head = nn.Dense(encoder_dim, num_classes)
        self.head.weight.set_data(initializer(initialization, [num_classes, encoder_dim]))
        self.stem = VitStem(encoder_dim, patch_size, image_size)
        if dropout:
            self.is_dropout = True
            self.dropout = nn.Dropout(keep_prob=(1. - dropout))
        self.encoder_input_mask = Tensor(np.ones((batch_size, seq_length, seq_length)),
                                         mstype.float16)
        self.return_feature = return_feature
    def init_weights(self, param_dict):
        """Full model weights initialization."""
        net_not_load = load_param_into_net(self, param_dict)
        return net_not_load

    def construct(self, img):
        tokens, _ = self.stem(img)
        tokens = self.cat((self.cls_token, tokens))
        tokens = self.add(tokens, self.encoder_pos_embedding)

        if self.is_dropout:
            temp = self.cast(tokens, mstype.float32)
            temp = self.dropout(temp)
            tokens = self.cast(temp, tokens.dtype)

        tokens = self.encoder(tokens, self.encoder_input_mask)[0]
        if self.norm is not None:
            tokens = self.norm(tokens)
        if self.return_feature:
            return tokens
        return self.head(tokens[:, 0])


class VitEval(MAEModule):
    """
    pass
    """

    def __init__(self,
                 batch_size,
                 patch_size,
                 image_size,
                 encoder_layers=12,
                 encoder_num_heads=12,
                 encoder_dim=768,
                 mlp_ratio=4,
                 num_classes=1001,
                 dropout=0.1,
                 initialization=XavierUniform()):
        super(VitEval, self).__init__(batch_size, image_size, patch_size)

        cls_token = Parameter(
            initializer(initialization, (1, 1, encoder_dim)),
            name='cls', requires_grad=True
        )
        self.cls_token = P.Tile()(cls_token, (batch_size, 1, 1))
        seq_length = self.num_patches + 1
        self.encoder_pos_embedding = Parameter(
            initializer(initialization, (1, seq_length, encoder_dim)),
            name='pos_embedding', requires_grad=True
        )
        parallel_config = ParallelConfig
        op_parallel_config = TransformerOpParallelConfig(data_parallel=parallel_config.dp,
                                                         model_parallel=parallel_config.mp,
                                                         pipeline_stage=parallel_config.pipeline_stage,
                                                         optimizer_shard=parallel_config.optimizer_shard,
                                                         vocab_emb_dp=parallel_config.vocab_emb_dp)
        print(f"TransformerEncoder batch_size: {batch_size}")
        self.encoder = TransformerEncoder(batch_size=batch_size, num_layers=encoder_layers,
                                          num_heads=encoder_num_heads, hidden_size=encoder_dim,
                                          ffn_hidden_size=encoder_dim * mlp_ratio, seq_length=seq_length,
                                          parallel_config=op_parallel_config)

        self.add = P.Add().shard(((parallel_config.dp, 1, 1), (1, 1, 1)))
        self.cast = P.Cast()
        self.cat = P.Concat(axis=1).shard(((parallel_config.dp, 1, 1),
                                           (parallel_config.dp, 1, 1)))
        self.norm = LayerNorm((encoder_dim,), parallel_config.dp)
        # self.head = nn.Dense(encoder_dim, num_classes)
        # self.head.weight.set_data(initializer(initialization, [num_classes, encoder_dim]))
        self.stem = VitStem(encoder_dim, patch_size, image_size)

        if dropout:
            self.is_dropout = True
            self.dropout = nn.Dropout(keep_prob=(1. - dropout))
            self.dropout.dropout.shard(((parallel_config.dp, 1, 1),))
        print("vit seq_length ===========>", batch_size, seq_length)
        self.encoder_input_mask = Tensor(np.ones((batch_size, seq_length, seq_length)),
                                         mstype.float16)
        self.print = P.Print()

    def init_weights(self, param_dict):
        """Full model weights initialization."""
        net_not_load = load_param_into_net(self, param_dict)
        return net_not_load

    def construct(self, img):
        tokens, _ = self.stem(img)
        # tokens = self.cat((self.cls_token, tokens))
        tokens = self.add(tokens, self.encoder_pos_embedding)

        if self.is_dropout:
            temp = self.cast(tokens, mstype.float32)
            temp = self.dropout(temp)
            tokens = self.cast(temp, tokens.dtype)

        tokens = self.encoder(tokens, self.encoder_input_mask)[0]
        if self.norm is not None:
            tokens = self.norm(tokens)

        return tokens



class VitEvalV2(MAEModule):
    """
    pass
    """

    def __init__(self,
                 batch_size,
                 patch_size,
                 image_size,
                 encoder_layers=12,
                 encoder_num_heads=12,
                 encoder_dim=768,
                 mlp_ratio=4,
                 channels=3,
                 dropout=0.1,
                 drop_path=0.1,
                 initialization=XavierUniform()):
        super(VitEvalV2, self).__init__(batch_size, image_size, patch_size)

        cls_token = Parameter(
            initializer(initialization, (1, 1, encoder_dim)),
            name='cls', requires_grad=True
        )
        self.cls_token = P.Tile()(cls_token, (batch_size, 1, 1))
        seq_length = self.num_patches + 1
        self.encoder_pos_embedding = Parameter(
            initializer(initialization, (1, seq_length, encoder_dim)),
            name='pos_embedding', requires_grad=True
        )

        # parallel_config = ParallelConfig
        # op_parallel_config = TransformerOpParallelConfig(data_parallel=parallel_config.dp,
        #                                                  model_parallel=parallel_config.mp,
        #                                                  pipeline_stage=parallel_config.pipeline_stage,
        #                                                  optimizer_shard=parallel_config.optimizer_shard,
        #                                                  vocab_emb_dp=parallel_config.vocab_emb_dp)
        print(f"TransformerEncoder batch_size: {batch_size}")
        self.encoder = TransformerEncoder(batch_size=batch_size, num_layers=encoder_layers,
                                          num_heads=encoder_num_heads, hidden_size=encoder_dim,
                                          ffn_hidden_size=encoder_dim * mlp_ratio, seq_length=seq_length,
                                          hidden_dropout_rate=drop_path)

        self.add = P.Add()
        self.cat = P.Concat(axis=1)
        self.stride_slice = P.StridedSlice()
        self.norm = _LayerNorm((encoder_dim,), eps=1e-6).to_float(mstype.float32)

        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size,
                                      in_features=channels, out_features=encoder_dim)

        self.dropout = _Dropout(keep_prob=(1. - dropout))
        print("vit seq_length ===========>", batch_size, seq_length)
        self.encoder_input_mask = Tensor(np.ones((batch_size, seq_length, seq_length)),
                                         mstype.float16)

    def init_weights(self, param_dict):
        """Full model weights initialization."""
        net_not_load = load_param_into_net(self, param_dict)
        return net_not_load

    def construct(self, img):

        tokens = self.patch_embed(img)
        tokens = self.cat((self.cls_token, tokens))
        tokens = self.add(tokens, self.encoder_pos_embedding)

        tokens = self.dropout(tokens)

        tokens = self.encoder(tokens, self.encoder_input_mask)[0]

        tokens = self.norm(tokens)

        return tokens

if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE, device_id=7)
    sdn = P.StandardNormal(seed=2)
    image = sdn((2, 3, 64, 64))
    model = VitEval(batch_size=2,
                    patch_size=8,
                    image_size=64)
    outs = model(image)
    print("suceesss", outs.shape)
