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

import mindspore.nn as nn

class CustomWithLossCell(nn.Cell):
    def __init__(self, model, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)

        self._model = model
        self._loss_fn = loss_fn

    def construct(self, x, is_training=True):
        output = self._model(x, is_training)
        x_rec = output['x_rec']
        loss = self._loss_fn(x_rec, x) + output['vq_output']['loss']
        return loss

def get_modeloss(opt):
    model_name = opt['model']['name'].lower()
    loss_name = opt['loss']['name'].lower()

    if model_name in ['vqvae', 'vqvae_wbn'] and loss_name == 'nmse':
        return CustomWithLossCell
    else:
        raise NotImplementedError(f"{model_name} + {loss_name}")