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
"""
 build optimizer for ms
 for params containing the words 'layernorm' and not containing 'bias', we choose the adam.
 for the other params, they are optimized by adam.
"""
from mindspore.nn.optim.adam import Adam, AdamWeightDecay


def build_optimizer(model, optim, betas, lr):
    """

    :param model:
    :param opts:
    :param lr:
    :return: optimizer
    """

    decay_filter = lambda x: 'layernorm' not in x.name.lower() and "bias" not in x.name.lower()
    param_optimizer = model.trainable_params()
    decay_params = list(filter(decay_filter, param_optimizer))
    other_params = list(filter(lambda x: not decay_filter(x), param_optimizer))
    group_params = [{
        'params': decay_params,
        'weight_decay': 1e-2
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': param_optimizer
    }]
    # currently Adam only
    if optim == 'adam':
        OptimCls = Adam
    elif optim == 'adamw':
        OptimCls = AdamWeightDecay
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(group_params,
                         learning_rate=lr, beta1=betas[0], beta2=betas[1])
    return optimizer
