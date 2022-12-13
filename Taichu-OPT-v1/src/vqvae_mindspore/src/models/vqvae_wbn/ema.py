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

from mindspore import ms_function
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype

@ms_function
def update_counter(c):
    return c + 1

@ms_function
def update_hidden(hidden, value, decay):
    return hidden - (hidden - value) * (1 - decay)

@ms_function
def update_average(hidden, counter, decay):
    return hidden / (1. - ops.Pow()(decay, counter))


class ExponentialMovingAverage(nn.Cell):
    """Maintains an exponential moving average for a value.

          Note this module uses debiasing by default. If you don't want this please use
          an alternative implementation.

          This module keeps track of a hidden exponential moving average that is
          initialized as a vector of zeros which is then normalized to give the average.
          This gives us a moving average which isn't biased towards either zero or the
          initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)

          Initially:

              hidden_0 = 0

          Then iteratively:

              hidden_i = (hidden_{i-1} - value) * (1 - decay)
              average_i = hidden_i / (1 - decay^i)

          Attributes:
            average: Variable holding average. Note that this is None until the first
              value is passed.
          """

    def __init__(self, decay, target, dtype):
        """Creates a debiased moving average module.
        Args:
          decay: The decay to use. Note values close to 1 result in a slow decay
            whereas values close to 0 result in faster decay, tracking the input
            values more closely.
          name: Name of the module.
        """
        super(ExponentialMovingAverage, self).__init__()
        self._pow = ops.Pow()

        self._decay = decay
        self._dtype = dtype
        zeros = ops.Zeros()

        self._counter = Parameter(Tensor(0, dtype=mstype.int32), requires_grad=False)
        self._hidden = Parameter(zeros(target.shape, dtype), requires_grad=False)
        self.average = Parameter(zeros(target.shape, dtype), requires_grad=False)

        # Tensor
        self._1 = Tensor(1, mstype.int32)

    def construct(self, value):
        self._counter = update_counter(self._counter)
        self._hidden = update_hidden(self._hidden, value, self._decay)
        self.average = update_average(self._hidden, self._counter, self._decay)
        return self.average

    def value(self):
        return self.average



if __name__ == '__main__':
    decay = 0.9
    shape = (5, 2)
    zeros = ops.Zeros()
    dtype = mstype.float32

    value = ops.StandardNormal(seed=2)(shape).astype(mstype.float32)
    print(f"Value: {value}")

    target = zeros(shape, dtype)
    test_ema = ExponentialMovingAverage(decay, target, dtype)

    cur_v = test_ema.value()
    print(f"0: {cur_v},  {Tensor(cur_v)}")

    for i in range(1, 10):
        print(f"Updating...")
        value = ops.StandardNormal(seed=2)(shape).astype(mstype.float32)
        test_ema.update(value)

        cur_v = test_ema.value()
        print(f"{i}: {cur_v},  \n{Tensor(cur_v)}")