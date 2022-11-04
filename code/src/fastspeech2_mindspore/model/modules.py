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
"""
fastspeech2 modules
"""
import os
import json
from collections import OrderedDict

import mindspore.numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from ...config import config
from ..utils.tools import pad

def bucketize(input1, boundary, right=True):
    right = not right
    # input1 = input1.asnumpy()
    # boundary = boundary.asnumpy()
    input1 = input1.astype(ms.float32)
    boundary = boundary.astype(ms.float32)
    output = np.digitize(input1, boundary, right=right)
    # output = Tensor(output)
    return output


class VarianceAdaptor(nn.Cell):
    """ Variance Adaptor """

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.exp = ops.Exp()
        self.linspace = ops.LinSpace()

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = ms.Parameter(
                self.exp(
                    self.linspace(Tensor(np.log(pitch_min), ms.float32), Tensor(np.log(pitch_max), ms.float32),
                                  n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = ms.Parameter(
                self.linspace(Tensor(pitch_min, ms.float32), Tensor(pitch_max, ms.float32), n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = ms.Parameter(
                self.exp(
                    self.linspace(Tensor(np.log(energy_min), ms.float32), Tensor(np.log(energy_max), ms.float32),
                                  n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = ms.Parameter(
                self.linspace(Tensor(energy_min, ms.float32), Tensor(energy_max, ms.float32), n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

        self.defalut_value = Tensor(1.0, dtype=ms.float32)
        self.round = ops.Round()
        self.select = ops.Select()
        self.tile = ops.Tile()
        self.unsqueezee = ops.ExpandDims()

        self.max_mel_len = config.MAX_MEL_LEN

        self.stride_slice_1 = ops.StridedSlice().shard(((1,),))
        self.add = ops.Add()

    def bucketize(self, x, boundary):
        x = x.astype(ms.float32)
        # minimum = boundary[0]
        # maximum = boundary[-1]
        minimum = self.stride_slice_1(boundary, (0,), (1,), (1,))
        maximum = self.stride_slice_1(boundary, (boundary.shape[0]-1,), (boundary.shape[0],), (1,))

        frac = (maximum - minimum) / boundary.shape[0]
        min_cond = x < minimum
        max_cond = x > maximum
        value = self.round((x - minimum) / frac).astype(ms.int32)
        minimum_array = self.tile(ops.scalar_to_array(0), x.shape)
        maximum_array = self.tile(ops.scalar_to_array(boundary.shape[0]-1), x.shape)
        min_output = self.select(min_cond, minimum_array, value)
        output = self.select(max_cond, maximum_array, min_output)
        return output

    def get_pitch_embedding(self, x, target, mask, control):
        """get_pitch_embedding"""
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(self.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                self.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding.astype(x.dtype)

    def get_energy_embedding(self, x, target, mask, control):
        """get_energy_embedding"""
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(self.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                self.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding.astype(x.dtype)

    def get_mask_from_lengths(self, lengths, max_len):
        ids = ops.tuple_to_array(ops.make_range(max_len)).astype(ms.int32)
        ids = self.tile(self.unsqueezee(ids, 0), (lengths.shape[0], 1))
        lengths = self.unsqueezee(lengths, 1)
        lengths = self.tile(lengths, (1, max_len))
        mask = (ids>=lengths)
        return mask

    def construct(
            self,
            x,
            src_mask,
            mel_mask=None,
            max_len=None,
            pitch_target=None,
            energy_target=None,
            duration_target=None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
    ):

        """VarianceAdapter construct"""
        log_duration_prediction = self.duration_predictor(x, src_mask)

        # if self.pitch_feature_level == "phoneme_level":
        #     pitch_prediction, pitch_embedding = self.get_pitch_embedding(
        #         x, pitch_target, src_mask, p_control
        #     )
        #     x = x + pitch_embedding
        # if self.energy_feature_level == "phoneme_level":
        #     energy_prediction, energy_embedding = self.get_energy_embedding(
        #         x, energy_target, src_mask, p_control
        #     )
        #     x = x + energy_embedding

        pitch_prediction, pitch_embedding = self.get_pitch_embedding(
            x, pitch_target, src_mask, p_control
        )
        x = x + pitch_embedding

        energy_prediction, energy_embedding = self.get_energy_embedding(
            x, energy_target, src_mask, p_control
        )
        x = x + energy_embedding

        # x, mel_len = self.length_regulator(x, duration_target, max_len)
        # duration_rounded = duration_target

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = ops.clip_by_value(
                (self.round(self.exp(log_duration_prediction) - 1) * d_control),
                clip_value_min=Tensor(0, ms.float32), clip_value_max=Tensor(float('inf'), ms.float32)
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = self.get_mask_from_lengths(mel_len, self.max_mel_len)

        # if self.pitch_feature_level == "frame_level":
        #     pitch_prediction, pitch_embedding = self.get_pitch_embedding(
        #         x, pitch_target, mel_mask, p_control
        #     )
        #     x = x + pitch_embedding
        #
        # if self.energy_feature_level == "frame_level":
        #     energy_prediction, energy_embedding = self.get_energy_embedding(
        #         x, energy_target, mel_mask, p_control
        #     )
        #     x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Cell):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()
        self.cat = ops.Concat(axis=0)
        self.mel_len = config.MAX_MEL_LEN
        self.src_len = config.MAX_SRC_LEN
        self.zeros = ops.Zeros()
        self.concat1 = ops.Concat(axis=1)
        self.gather = ops.GatherD()
        self.tile = ops.Tile()
        self.cumsum = ops.CumSum()

    def LR_GRAPH(self, x, duration):
        index = ops.tuple_to_array(ops.make_range(self.mel_len)).view(1, self.mel_len, 1)
        index = index.astype(ms.int32)
        redundant_vec = self.zeros((x.shape[0], 1, x.shape[2]), x.dtype)
        cum_duration = self.cumsum(duration, -1).view(duration.shape[0], 1, duration.shape[-1])
        gather_index = (cum_duration<=index).astype(ms.float32).sum(-1)

        enhanced_x = self.concat1((x, redundant_vec))
        index = self.tile(gather_index.view(gather_index.shape + (1,)), (1, 1, x.shape[-1]))
        output = self.gather(enhanced_x, 1, index.astype(ms.int32))
        mel_lens = duration.astype(ms.float32).sum(1)
        return output, mel_lens.astype(ms.int32)

    def LR(self, x, duration, max_len):
        """LengthRegulator"""
        output = []
        mel_len = []

        for i in range(x.shape[0]):
            batch = x[i]
            expand_target = duration[i]
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, Tensor(mel_len)

    def expand(self, batch, predicted):
        """LengthExpand"""
        out = []
        for i in range(batch.shape[0]):
            vec = batch[i]
            expand_size = predicted[i]
            if expand_size > 0:
                # size_0 = expand_size.asnumpy().tolist()
                size_0 = 1
                vec = np.expand_dims(vec, 0)
                size_1 = vec.shape[1]
                vec = np.broadcast_to(vec, (size_0, size_1))
                out.append(vec)
        out = self.cat(out)

        return out

    def construct(self, x, duration, max_len):
        """LengthRegulator construct"""
        # output, mel_len = self.LR(x, duration, max_len)
        output, mel_len = self.LR_GRAPH(x, duration)
        return output, mel_len


class VariancePredictor(nn.Cell):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.SequentialCell(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm((self.filter_size,))),
                    ("dropout_1", nn.Dropout(1 - self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm((self.filter_size,))),
                    ("dropout_2", nn.Dropout(1 - self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Dense(self.conv_output_size, 1)

    # encoder_output: batch_size * seq_len * dim_size
    def construct(self, encoder_output, mask):
        """ construct"""
        out = self.conv_layer(encoder_output)

        out = self.linear_layer(out)

        # out: batch_size * seq_len
        out = out.squeeze(-1)

        # if mask is not None:
        #     out = out.masked_fill(mask.to(ms.bool_), 0.0)

        return out


class Conv(nn.Cell):
    """
    Convolution Module
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        # in_channels,
        # out_channels,
        # kernel_size,
        # stride = 1,
        # pad_mode = 'same',
        # padding = 0,
        # dilation = 1,
        # group = 1,
        # has_bias = False,
        # weight_init = 'normal',
        # bias_init = 'zeros'

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            has_bias=bias,
            pad_mode="pad",
        )

    def construct(self, x):
        x = x.transpose(0, 2, 1)
        x = self.conv(x)
        x = x.transpose(0, 2, 1)

        return x
