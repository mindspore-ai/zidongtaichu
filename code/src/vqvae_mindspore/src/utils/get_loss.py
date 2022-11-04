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

import os
try:
    from src.loss.nmse import nMSELoss
except:
    from src.vqvae_mindspore.src.loss.nmse import nMSELoss

def get_loss(loss_opt):
    name = loss_opt['name'].lower()

    if name == 'nmse':
        loss = nMSELoss(reduction='mean')
    else:
        raise ValueError(f"!!!!! No implementation for loss {name} !!!!!")

    return loss