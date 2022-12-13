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

try:
    from src.models.vqvae_wbn.vqvae import VQVAEModel as VQVAE_wBN
except:
    from src.vqvae_mindspore.src.models.vqvae_wbn.vqvae import VQVAEModel as VQVAE_wBN

def get_model(model_opt):
    model_name = model_opt['name'].lower()
    if model_name == 'vqvae_wbn':
        model = VQVAE_wBN(num_hiddens=model_opt['num_hiddens'],
                           num_residual_layers=model_opt['num_residual_layers'],
                           num_residual_hiddens=model_opt['num_residual_hiddens'],
                           embedding_dim=model_opt['embedding_dim'],
                           num_embeddings=model_opt['num_embeddings'],
                           downsample=model_opt['downsample'],
                           commitment_cost=model_opt['commitment_cost'],
                           decay=model_opt['decay'])
    else:
        raise ValueError(f"!!!!! No Implementation for {model_name} !!!!!")

    return model