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

import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F

try:
    from src.models.vqvae_wbn.vq_ema import VectorQuantizerEMA
    from src.models.vqvae_wbn.encoder import Encoder
    from src.models.vqvae_wbn.decoder import Decoder
except:
    from src.vqvae_mindspore.src.models.vqvae_wbn.vq_ema import VectorQuantizerEMA
    from src.vqvae_mindspore.src.models.vqvae_wbn.encoder import Encoder
    from src.vqvae_mindspore.src.models.vqvae_wbn.decoder import Decoder

class VQVAEModel(nn.Cell):
    def __init__(self, num_hiddens, num_residual_layers,
                 num_residual_hiddens, downsample,
                 embedding_dim, num_embeddings,
                 commitment_cost, decay, data_variance=0.0632704):
        super(VQVAEModel, self).__init__()
        self.prt = ops.Print()
        self._is_training = False

        self._encoder = Encoder(num_hiddens, num_residual_layers,
                                num_residual_hiddens, downsample)
        self._decoder = Decoder(num_hiddens, num_residual_layers,
                                num_residual_hiddens, downsample)

        self._relu = nn.ReLU()
        self._pre_vq = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1,
                                 pad_mode='same', has_bias=False)
        self._suf_vq = nn.Conv2d(embedding_dim, num_hiddens, kernel_size=1,
                                 pad_mode='same', has_bias=False)

        self._vq_ema = VectorQuantizerEMA(embedding_dim, num_embeddings,
                                          commitment_cost, decay)
        self._data_variance = data_variance

    def get_xrec_from_codes(self, codes):
        # codes: [b, ch, cw]
        quantize = self._vq_ema.quantize_code(codes)
        h = self._suf_vq(quantize)
        x_rec = self._decoder.construct(h)
        return x_rec

    def construct(self, inputs, is_training):
        h = self._encoder.construct(inputs)
        z = self._pre_vq(h)
        vq_output = self._vq_ema.construct(z, is_training=is_training)
        h = self._suf_vq(vq_output['quantize'])
        x_rec = self._decoder.construct(h)

        return {
            'x_rec': x_rec,
            'vq_output': vq_output
        }

if __name__ == '__main__':
    from mindspore import context
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    num_hiddens = 8
    num_residual_layers = 2
    num_residual_hiddens = 4
    embedding_dim, num_embeddings = 4, 16
    commitment_cost, decay = 1, 0.9

    model = VQVAEModel(num_hiddens, num_residual_layers, num_residual_hiddens, 4,
                       embedding_dim, num_embeddings, commitment_cost, decay)
    model._is_training = True

    for i in range(10):
        # ipdb.set_trace()
        inputs = Tensor(np.ones((2, 3, 16, 16)), dtype=mstype.float32)
        x_rec = model.construct(inputs, False)
        loss = ((x_rec - inputs)**2).mean()
        print(f"Iter:{i},  loss:{loss}")