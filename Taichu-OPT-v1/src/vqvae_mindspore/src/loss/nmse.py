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

import mindspore
import mindspore.ops as ops
from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype
from mindspore.nn import LossBase, MSELoss
from mindspore.ops import functional as F

class nMSELoss(LossBase):
    def __init__(self, reduction='mean', data_variance=0.0632704):
        super(nMSELoss, self).__init__(reduction)

        self.pow = ops.Pow()
        self.cast = ops.Cast()
        self.prt = ops.Print()
        self.data_variance = data_variance

    def construct(self, output, target):
        xrec = self.cast(output, mindspore.float32)
        x = self.cast(target, mindspore.float32)

        loss = F.square(xrec - x)
        loss = self.get_loss(loss)
        loss = F.div(loss, self.data_variance)
        return loss
