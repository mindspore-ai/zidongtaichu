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
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.common.dtype as mstype

try:
    from src.models.vqvae_wbn.residualstack import ResidualStack
except:
    from src.vqvae_mindspore.src.models.vqvae_wbn.residualstack import ResidualStack

class Decoder(nn.Cell):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, downsample=4):
        super(Decoder, self).__init__()

        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layer_pre = nn.SequentialCell([
            nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, pad_mode='same', has_bias=True),
            nn.BatchNorm2d(num_hiddens),
            nn.ReLU(),
        ])

        self._append_modules = None
        if downsample == 8:
            self._append_modules = nn.SequentialCell([
                nn.Conv2dTranspose(self._num_hiddens, self._num_hiddens,
                                                      kernel_size=6, stride=2, padding=2,
                                                      pad_mode='pad', has_bias=True),
                nn.BatchNorm2d(self._num_hiddens),
                nn.ReLU(),
            ])

        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens
        )

        self._layer2 = nn.SequentialCell([
            nn.Conv2dTranspose(self._num_hiddens, self._num_hiddens // 2,
                                          kernel_size=6, stride=2, padding=2,
                                          pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(self._num_hiddens // 2),
            nn.ReLU(),
        ])
        self._layer3 = nn.Conv2dTranspose(self._num_hiddens // 2, 3,
                                          kernel_size=6, stride=2, padding=2,
                                          pad_mode='pad', has_bias=True)
        self._relu = nn.ReLU()


    def construct(self, inputs):
        h1 = self._layer_pre(inputs)
        if self._append_modules is not None:
            h1 = self._append_modules(h1)
        h2 = self._residual_stack(h1)
        h3 = self._layer2(h2)
        x = self._layer3(h3)
        return x

if __name__ == '__main__':
    num_hiddens = 8
    num_residual_layers = 2
    num_residual_hiddens = 4
    dtype = mstype.float32
    inputs = Tensor(np.random.randn(2, 8, 4, 4), dtype=dtype)
    print(f"Input shape: {inputs.shape}")

    decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens, downsample=4)
    outputs = decoder.construct(inputs)
    print(f"Output shape (ds4): {outputs.shape}")

    decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens, downsample=8)
    outputs = decoder.construct(inputs)
    print(f"Output shape (ds8): {outputs.shape}")


