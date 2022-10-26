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
"""Transformer beam search module."""

import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor

from .parallel_transformer import ParallelConfig
INF = 1. * 1e9


class LengthPenalty(nn.Cell):
    """
    Normalize scores of translations according to their length.

    Args:
        weight (float): Weight of length penalty. Default: 1.0.
        compute_type (:class:`mindspore.dtype`): Compute type in Transformer. Default: mstype.float32.
    """

    def __init__(self,
                 weight=1.0,
                 parallel_config=ParallelConfig,
                 compute_type=mstype.float32):
        super(LengthPenalty, self).__init__()
        self.cast = P.Cast()
        self.five = Tensor(5.0, mstype.float32)
        self.six = Tensor(6.0, mstype.float32)
        self.weight = Tensor(weight, mstype.float32)
        self.add = P.Add().shard(((parallel_config.dp, 1), ()))
        self.div = P.RealDiv().shard(((parallel_config.dp, 1), ()))
        self.pow = P.Pow().shard(((parallel_config.dp, 1), ()))

    def construct(self, length_tensor):
        length_tensor_ = self.cast(length_tensor, mstype.float32)
        output = self.add(length_tensor_, self.five)
        output = self.div(output, self.six)
        output = self.pow(output, self.weight)
        return output

class TileBeam(nn.Cell):
    """
    TileBeam.

    Args:
        beam_width (int): beam width setting. Default: 4.
        compute_type (:class:`mindspore.dtype`): Compute type in Transformer. Default: mstype.float32.
    """

    def __init__(self,
                 beam_width,
                 parallel_config=ParallelConfig,
                 compute_type=mstype.float32):
        super(TileBeam, self).__init__()
        self.beam_width = beam_width
        self.expand = P.ExpandDims().shard(((parallel_config.dp, 1, 1, 1),))
        self.tile = P.Tile().shard(((parallel_config.dp, 1, 1, 1, 1),))
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, input_tensor):
        """
        input_tensor: shape [batch, dim1, dim2, dim3]
        output_tensor: shape [batch, beam, dim1, dim2, dim3]
        """
        shape = self.shape(input_tensor)
        input_tensor = self.expand(input_tensor, 1)
        tile_shape = (1,) + (self.beam_width,)
        for _ in range(len(shape) - 1):
            tile_shape = tile_shape + (1,)
        output = self.tile(input_tensor, tile_shape)
        out_shape = (shape[0] * self.beam_width,) + shape[1:]
        output = self.reshape(output, out_shape)
        return output

class TileBeam1(nn.Cell):
    """
    TileBeam.

    Args:
        beam_width (int): beam width setting. Default: 4.
        compute_type (:class:`mindspore.dtype`): Compute type in Transformer. Default: mstype.float32.
    """

    def __init__(self,
                 beam_width,
                 parallel_config=ParallelConfig,
                 compute_type=mstype.float32):
        super(TileBeam1, self).__init__()
        self.beam_width = beam_width
        self.expand = P.ExpandDims().shard(((parallel_config.dp, 1, 1),))
        self.tile = P.Tile().shard(((parallel_config.dp, 1, 1, 1),))
        self.reshape = P.Reshape()
        self.shape = P.Shape()
    
    def construct(self, input_tensor):
        """
        input_tensor: shape [batch, dim1, dim2]
        output_tensor: shape [batch, beam, dim1, dim2]
        """
        shape = self.shape(input_tensor)
        input_tensor = self.expand(input_tensor, 1)
        tile_shape = (1,) + (self.beam_width,)
        for _ in range(len(shape) - 1):
            tile_shape = tile_shape + (1,)
        output = self.tile(input_tensor, tile_shape)
        out_shape = (shape[0] * self.beam_width,) + shape[1:]
        output = self.reshape(output, out_shape)
        return output

class Mod(nn.Cell):
    """
    Mod function.

    Args:
        compute_type (:class:`mindspore.dtype`): Compute type in Transformer. Default: mstype.float32.
    """
    def __init__(self,
                 compute_type=mstype.float32):
        super(Mod, self).__init__()
        self.compute_type = compute_type
        self.floor_div = P.FloorDiv()
        self.sub = P.Sub()
        self.multiply = P.Mul()

    def construct(self, input_x, input_y):
        x = self.floor_div(input_x, input_y)
        x = self.multiply(x, input_y)
        x = self.sub(input_x, x)
        return x

class BeamSearchDecoder(nn.Cell):
    """
    Beam search decoder.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input sequence.
        vocab_size (int): Size of vocabulary.
        decoder (:class:`TransformerDecoderStep`): Decoder module.
        beam_width (int): beam width setting. Default: 4.
        length_penalty_weight (float): Weight of length penalty. Default: 1.0.
        max_decode_length (int): max decode length. Default: 128.
        sos_id (int): Id of sequence start token. Default: 1.
        eos_id (int): Id of sequence end token. Default: 2.
        compute_type (:class:`mindspore.dtype`): Compute type in Transformer. Default: mstype.float32.
    """

    def __init__(self,
                 batch_size,
                 seq_length,
                 vocab_size,
                 decoder,
                 beam_width=4,
                 length_penalty_weight=1.0,
                 max_decode_length=150,
                 sos_id=1,
                 eos_id=2,
                 parallel_config=ParallelConfig,
                 compute_type=mstype.float32,
                 task=""):
        super(BeamSearchDecoder, self).__init__(auto_prefix=False)
        self.task=task
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.beam_width = beam_width
        self.length_penalty_weight = length_penalty_weight
        self.max_decode_length = max_decode_length
        self.decoder = decoder

        self.expand = P.ExpandDims().shard(((1, 1),))
        self.reshape = P.Reshape()
        self.shape_flat = (-1,)
        self.shape = P.Shape()
        self.multinomial = ops.Multinomial()
        self.multinomial.add_prim_attr("primitive_target", "CPU")

        self.zero_tensor = Tensor(np.zeros([batch_size, beam_width]), mstype.float32)
        self.ninf_tensor = Tensor(np.full([batch_size, beam_width], -INF), mstype.float32)

        self.flat_shape = (batch_size, beam_width * vocab_size)
        self.vocab_size_tensor = Tensor(self.vocab_size, mstype.int32)
        self.eos_ids = Tensor(np.full([batch_size, beam_width], eos_id), mstype.int32)

        beam_ids = np.tile(np.arange(beam_width).reshape((1, beam_width)), [batch_size, 1])
        self.beam_ids = Tensor(beam_ids, mstype.int32)
        batch_ids = np.arange(batch_size * beam_width).reshape((batch_size, beam_width)) // beam_width
        self.batch_ids = Tensor(batch_ids, mstype.int32)
        self.concat = P.Concat(axis=-1).shard(((1, 1, 1), (1, 1, 1)))
        self.gather_nd2 = P.GatherNd().shard(((1, 1), (parallel_config.dp, 1, 1)))
        self.gather_nd3 = P.GatherNd().shard(((1, 1, 1), (parallel_config.dp, 1, 1)))
        self.select = P.Select().shard(((parallel_config.dp, 1), (parallel_config.dp, 1), (parallel_config.dp, 1)))
        self.strided_slice3 = P.StridedSlice().shard(((parallel_config.dp, 1, 1),))

        self.add = P.Add().shard(((parallel_config.dp, 1), ()))
        self.add1 = P.Add().shard(((parallel_config.dp, 1), (parallel_config.dp, 1)))
        self.add2 = P.Add().shard(((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.sub = P.Sub().shard(((parallel_config.dp, 1), ()))
        self.sub1 = P.Sub().shard(((parallel_config.dp, 1), (parallel_config.dp, 1)))
        self.mul = P.Mul().shard(((parallel_config.dp, 1), ()))
        self.div1 = P.RealDiv().shard(((parallel_config.dp, 1), (parallel_config.dp, 1)))
        self.greater_equal = P.GreaterEqual().shard(((parallel_config.dp, 1), ()))
        self.equal = P.Equal().shard(((parallel_config.dp, 1), (parallel_config.dp, 1)))
        self.topk = P.TopK(sorted=True).shard(((parallel_config.dp, 1),))
        self.zeroslike = P.ZerosLike().shard(((parallel_config.dp, 1),))
        self.cast = P.Cast()

        # init inputs and states
        self.start_ids = Tensor(np.full([batch_size * beam_width, 1], sos_id), mstype.int32)
        self.init_seq = Tensor(np.full([batch_size, beam_width, 1], sos_id), mstype.int32)
        init_scores = np.tile(np.array([[0.] + [-INF] * (beam_width - 1)]), [batch_size, 1])
        self.init_scores = Tensor(init_scores, mstype.float32)
        self.init_finished = Tensor(np.zeros([batch_size, beam_width], dtype=np.bool))
        self.init_length = Tensor(np.zeros([batch_size, beam_width], dtype=np.int32))
        self.length_penalty = LengthPenalty(weight=length_penalty_weight, parallel_config=parallel_config)
        self.one = Tensor(1, mstype.int32)
        self.tokens = Tensor(np.array(range(self.vocab_size)).reshape((vocab_size, 1, 1)), mstype.int32)
        self.start_ids_max = Tensor(np.full([batch_size * beam_width, self.max_decode_length], sos_id), mstype.int32)
        self.scalar_to_tensor = P.ScalarToTensor()
        self.scatter_update = P.ScatterUpdate()

    def one_step(self, cur_input_ids, enc_states, enc_attention_mask, state_log_probs,
                 state_seq, state_finished, state_length):
        """
        One step for decode
        """
        # log_probs = self.decoder(cur_input_ids, enc_states, enc_attention_mask, self.seq_length)
        log_probs = self.decoder(cur_input_ids, enc_states, enc_attention_mask)
        log_probs = self.reshape(log_probs, (self.batch_size, self.beam_width, self.vocab_size))

        # select topk indices
        total_log_probs = self.add2(log_probs, self.expand(state_log_probs, -1))

        # mask finished beams
        mask_tensor = self.select(state_finished, self.ninf_tensor, self.zero_tensor)
        total_log_probs = self.add2(total_log_probs, self.expand(mask_tensor, -1))

        # reshape scores to [batch, beam*vocab]
        flat_scores = self.reshape(total_log_probs, self.flat_shape)
        # select topk
        topk_scores, topk_indices = self.topk(flat_scores, self.beam_width)

        temp = topk_indices
        beam_indices = self.zeroslike(topk_indices)
        for _ in range(self.beam_width - 1):
            temp = self.sub(temp, self.vocab_size_tensor)
            res = self.cast(self.greater_equal(temp, 0), mstype.int32)
            beam_indices = self.add1(beam_indices, res)
        word_indices = self.sub1(topk_indices, self.mul(beam_indices, self.vocab_size_tensor))
        # ======================================================================

        # mask finished indices
        beam_indices = self.select(state_finished, self.beam_ids, beam_indices)
        word_indices = self.select(state_finished, self.eos_ids, word_indices)
        topk_scores = self.select(state_finished, state_log_probs, topk_scores)

        ###### put finished sequences to the end
        # sort according to scores with -inf for finished beams
        tmp_log_probs = self.select(
            self.equal(word_indices, self.eos_ids),
            self.ninf_tensor,
            topk_scores)
        _, tmp_indices = self.topk(tmp_log_probs, self.beam_width)
        # update
        tmp_gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(tmp_indices, -1)))
        beam_indices = self.gather_nd2(beam_indices, tmp_gather_indices)
        word_indices = self.gather_nd2(word_indices, tmp_gather_indices)
        topk_scores = self.gather_nd2(topk_scores, tmp_gather_indices)

        ###### generate new beam_search states
        # gather indices for selecting alive beams
        gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(beam_indices, -1)))

        # length add 1 if not finished in the previous step
        length_add = self.add(state_length, self.one)
        state_length = self.select(state_finished, state_length, length_add)
        state_length = self.gather_nd2(state_length, gather_indices)

        # concat seq
        seq = self.gather_nd3(state_seq, gather_indices)
        state_seq = self.concat((seq, self.expand(word_indices, -1)))

        # new finished flag and log_probs
        state_finished = self.equal(word_indices, self.eos_ids)
        state_log_probs = topk_scores

        ###### generate new inputs and decoder states
        cur_input_ids = self.reshape(state_seq, (self.batch_size * self.beam_width, -1))
        return cur_input_ids, state_log_probs, state_seq, state_finished, state_length

    def construct(self, enc_states, enc_attention_mask):
        if self.task == "T2I": # For T2I, only sample from topk probable tokens for each position
            topk = 200
            temperature = 1
            cur_input_ids = self.start_ids_max
            for index in range(self.max_decode_length - 1):
                log_probs = self.decoder(cur_input_ids, enc_states, enc_attention_mask, index=index + 1)
                topk_scores, topk_indices = self.topk(log_probs, topk)
                topk_scores = P.Softmax()(topk_scores / temperature)
                select_ind = topk_indices[0, self.multinomial(topk_scores, 1)[0, 0]]
                select_ind = self.tokens[select_ind]

                if index != self.max_decode_length - 2:
                    tmp_cur_input_ids = self.concat((cur_input_ids[:, :index + 1], select_ind))
                    cur_input_ids = self.concat((tmp_cur_input_ids, cur_input_ids[:, index + 2:]))
                else:
                    cur_input_ids = self.concat((cur_input_ids[:, :index + 1], select_ind))

            predicted_ids = cur_input_ids[:, 1:]
            return predicted_ids
        else:
            """Get beam search result."""
            cur_input_ids = self.start_ids
            # beam search states
            state_log_probs = self.init_scores
            state_seq = self.init_seq
            state_finished = self.init_finished
            state_length = self.init_length

            for _ in range(self.max_decode_length):
                # run one step decoder to get outputs of the current step
                # shape [batch*beam, 1, vocab]
                cur_input_ids, state_log_probs, state_seq, state_finished, state_length = self.one_step(
                    cur_input_ids, enc_states, enc_attention_mask, state_log_probs, state_seq, state_finished, state_length)
            # add length penalty scores
            penalty_len = self.length_penalty(state_length)
            # get penalty length
            log_probs = self.div1(state_log_probs, penalty_len)

            # sort according to scores
            _, top_beam_indices = self.topk(log_probs, self.beam_width)
            gather_indices = self.concat((self.expand(self.batch_ids, -1), self.expand(top_beam_indices, -1)))
            # sort sequence
            predicted_ids = self.gather_nd3(state_seq, gather_indices)
            # take the first one
            ids_shape = self.shape(predicted_ids)
            predicted_ids = self.strided_slice3(predicted_ids, (0, 0, 0), (ids_shape[0], 1, ids_shape[2]), (1, 1, 1))
            # predicted_ids = predicted_ids[::, 0:1:1, ::]
            return predicted_ids
