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

def conv(in_channels, out_channels, k, s=1, p=0, m="pad", b=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p,
                     has_bias=b, pad_mode=m)

class Encoder(nn.Cell):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, downsample=4):
        super(Encoder, self).__init__()

        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layers = nn.SequentialCell([
            nn.Conv2d(3, self._num_hiddens // 2,
                      kernel_size=5, stride=2, padding=2,
                      pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(self._num_hiddens // 2),
            nn.ReLU(),
            nn.Conv2d(self._num_hiddens // 2, self._num_hiddens,
                      kernel_size=5, stride=2, padding=2,
                      pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(self._num_hiddens),
            nn.ReLU(),
            nn.Conv2d(self._num_hiddens, self._num_hiddens,
                      kernel_size=3, stride=1,
                      pad_mode='same', has_bias=True),
            nn.BatchNorm2d(self._num_hiddens),
            nn.ReLU(),
        ])

        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens
        )

        self._append_modules = None
        if downsample == 8:
            self._append_modules = nn.SequentialCell([
                nn.Conv2d(self._num_hiddens, self._num_hiddens,
                          kernel_size=5, stride=2, padding=2, pad_mode='pad', has_bias=True),
                nn.BatchNorm2d(self._num_hiddens),
                nn.ReLU(),
                nn.Conv2d(self._num_hiddens, self._num_hiddens,
                          kernel_size=3, stride=1, pad_mode='same', has_bias=True),
                nn.BatchNorm2d(self._num_hiddens),
                nn.ReLU(),
            ])

    def construct(self, inputs):
        h = self._layers(inputs)
        code = self._residual_stack(h)
        if self._append_modules is not None:
            code = self._append_modules(code)
        return code

if __name__ == '__main__':
    num_hiddens = 8
    num_residual_layers = 2
    num_residual_hiddens = 4
    dtype = mstype.float32
    inputs = Tensor(np.random.randn(2, 3, 16, 16), dtype=dtype)
    print(f"Input shape: {inputs.shape}")

    encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens, downsample=4)
    outputs = encoder.construct(inputs)
    print(f"Output shape (ds4): {outputs.shape}")

    encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens, downsample=8)
    outputs = encoder.construct(inputs)
    print(f"Output shape (ds8): {outputs.shape}")


