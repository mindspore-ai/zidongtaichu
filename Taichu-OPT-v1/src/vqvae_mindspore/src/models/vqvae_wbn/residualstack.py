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
import mindspore.ops as ops
from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype

class ResidualStack(nn.Cell):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()

        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._relu = nn.ReLU()
        self._layers = nn.CellList([])
        for i in range(num_residual_layers):
            curlayer = nn.SequentialCell([
                nn.Conv2d(num_hiddens, num_residual_hiddens,
                          kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=True),
                nn.BatchNorm2d(num_residual_hiddens),
                nn.ReLU(),
                nn.Conv2d(num_residual_hiddens, num_hiddens,
                          kernel_size=1, stride=1, padding=0, pad_mode='pad', has_bias=True),
                nn.BatchNorm2d(num_hiddens),
                nn.ReLU(),
            ])
            self._layers.append(curlayer)

    def construct(self, inputs):
        h = inputs
        for layer in self._layers:
            z = layer(h)
            h = h + z
        return self._relu(h)

if __name__ == '__main__':
    test_resi = ResidualStack(8, 2, 4)
    inputs = Tensor(np.random.randn(2, 8, 16, 16).astype(np.float32))

    print(f"IN shape: {inputs.shape}")
    out = test_resi.construct(inputs)
    print(f"OUT shape: {out.shape}")