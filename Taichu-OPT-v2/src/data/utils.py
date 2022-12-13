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
"""utils"""
import numpy as np
from src.config import config


def pad_sequence(sequences, batch_first=True, padding_value=0.0, max_lens=config.MAX_DEFAULT_LEN):
    """pad_sequence"""
    lens = [len(x) for x in sequences]
    if max_lens == -1:
        max_lens = max(lens)

    padded_seq = []
    for x in sequences:
        pad_width = [(0, max_lens - len(x))]
        # pad = [(padding_value, padding_value) for _ in range(1, len(sequences[0].shape))]
        # pad_width.extend(pad)
        padded_seq.append(np.pad(x, pad_width, constant_values=(padding_value, padding_value)))

    sequences = np.stack(padded_seq, axis=0 if batch_first else 1)
    return sequences


def masked_fill(x, mask, value):
    """masked_fill"""
    mask = np.broadcast_to(mask, x.shape)
    if mask.dtype != np.bool_:
        mask = mask == 1.0
    y = x.copy()
    y[mask] = value
    return y
