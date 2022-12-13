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
""" loss """

import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
import numpy as np
from mindspore.communication.management import get_group_size

class CrossEntropy(nn.Cell):
    """ CrossEntropy """

    def __init__(self, parallel_config):
        super(CrossEntropy, self).__init__()
        self.mean = ops.ReduceMean().shard(((1,),))
        self.sum = ops.ReduceSum().shard(((parallel_config.dp, 1),))
        self.onehot = ops.OneHot().shard(((parallel_config.dp, 1), (), ()))
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.log_softmax = ops.LogSoftmax().shard(((parallel_config.dp, 1),))
        self.neg = ops.Neg().shard(((parallel_config.dp, 1),))
        self.sub = ops.Sub().shard(((parallel_config.dp, 1), (parallel_config.dp, 1)))
        self.max = ops.ArgMaxWithValue(axis=-1, keep_dims=True).shard(((parallel_config.dp, 1),))
        self.mul = ops.Mul().shard(((parallel_config.dp, 1), (parallel_config.dp, 1)))

    def construct(self, logits, label):
        """ construct """

        flat_label = label.view(-1,)
        onehot_label = self.onehot(flat_label, logits.shape[-1], self.on_value, self.off_value)
        _, max_value = self.max(logits)
        logits = self.sub(logits, max_value)
        log_logits = self.log_softmax(logits)
        ce = self.neg(self.mul(log_logits, onehot_label))
        loss = self.sum(ce, -1)
        return self.mean(loss)


class MSE(nn.Cell):
    """ MSE """

    def __init__(self, parallel_config):
        super(MSE, self).__init__()
        self.mean = ops.ReduceMean().shard(((1, 1),))
        self.sub = ops.Sub().shard(((parallel_config.dp, 1), (parallel_config.dp, 1)))
        self.square = ops.Square().shard(((parallel_config.dp, 1),))

    def construct(self, logits, label):
        """ construct """
        square_error = self.square(self.sub(label, logits))
        return self.mean(square_error)


class ClipLoss(nn.Cell):
    def __init__(self, batch_size, parallel_config, temp=0.07):
        super(ClipLoss, self).__init__()

        print(f"clip temp: {temp}")

        self.clip_temp = Parameter(Tensor(np.log(1 / temp), mstype.float32))
        self.exp = ops.Exp()

        group_size = 1

        try:
            self.allgather = ops.AllGather()
            group_size = get_group_size()
        except Exception as e:
            print(e)

        self.bmm = ops.MatMul(transpose_a=False, transpose_b=True)
        # self.div = ops.Div().shard(((1, 1), (1, 1)))

        self.cat_0 = ops.Concat(axis=0)
        self.mean = ops.ReduceMean().shard(((1,),))

        self.cross_entropy = CrossEntropy(parallel_config)

        global_batch_size = batch_size * group_size
        self.clip_ground_truth = Tensor(np.arange(0, global_batch_size), mstype.int32)


    def construct(self, text_feat, image_feat):

        text_feat = self.allgather(text_feat)
        image_feat = self.allgather(image_feat)

        # N * N
        logits = self.bmm(image_feat.astype(mstype.float16), text_feat.astype(mstype.float16)).astype(text_feat.dtype)
        logits = logits * self.exp(self.clip_temp)

        loss_1 = self.cross_entropy(logits, self.clip_ground_truth)
        loss_2 = self.cross_entropy(logits.transpose(), self.clip_ground_truth)

        loss = self.cat_0((loss_1.view(1, ), loss_2.view(1, )))
        final_loss = self.mean(loss)

        return final_loss

class ContrastiveLoss(nn.Cell):
    """ ContrastiveLoss """

    def __init__(self, logit_temp, parallel_config):
        super(ContrastiveLoss, self).__init__()
        self.expand_dims = ops.ExpandDims().shard(((parallel_config.dp, 1),))
        self.concat = ops.Concat(axis=1).shard(((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.logit_temp = logit_temp
        self.squeeze = ops.Squeeze().shard(((parallel_config.dp, 1, 1),))
        self.div = ops.RealDiv().shard(((parallel_config.dp, 1), ()))
        self.div1 = ops.RealDiv().shard(((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.log_softmax = ops.LogSoftmax().shard(((parallel_config.dp, 1),))
        self.slice = ops.StridedSlice().shard(((parallel_config.dp, 1),))
        self.batch_matmul = ops.BatchMatMul(transpose_b=True).shard(
            ((parallel_config.dp, 1, 1), (parallel_config.dp, 1, 1)))
        self.sum = ops.ReduceSum(keep_dims=True).shard(((parallel_config.dp, 1, 1),))
        self.mean = ops.ReduceMean().shard(((1, 1),))
        self.square = ops.Square().shard(((parallel_config.dp, 1, 1),))
        self.sqrt = ops.Sqrt().shard(((parallel_config.dp, 1, 1),))
        self.sub = ops.Sub().shard(((), (parallel_config.dp, 1, 1)))
        self.neg = ops.Neg().shard(((parallel_config.dp, 1),))
        self.add = ops.Add().shard(((parallel_config.dp, 1, 1), ()))
        self.eps = 1e-7

    def construct(self, prediction, pos_sample, neg_samples):
        """ construct """

        pos_sample = self.expand_dims(pos_sample, 1)
        target = self.concat((pos_sample, neg_samples))
        prediction = self.expand_dims(prediction, 1)
        prediction = prediction.astype(mstype.float32)
        target = target.astype(mstype.float32)
        a = self.div1(prediction, self.add(self.sqrt(self.sum(self.square(prediction), -1)), self.eps))
        b = self.div1(target, self.add(self.sqrt(self.sum(self.square(target), -1)), self.eps))
        cos = self.batch_matmul(a, b)
        cos = self.sub(1, cos)
        logits = self.squeeze(cos)
        tempeture_logits = self.div(logits, self.logit_temp)
        log_logits = self.neg(self.log_softmax(tempeture_logits))
        pos_log_logits = self.slice(log_logits, (0, 0), (log_logits.shape[0], 1), (1, 1))
        return self.mean(pos_log_logits)


class TransformerTrainingLoss(nn.Cell):
    """
    Provide transformer training loss.

    Args:
        config (TransformerConfig): The config of Transformer.

    Returns:
        Tensor, total loss.
    """

    def __init__(self, config, parallel_config):
        super(TransformerTrainingLoss, self).__init__(auto_prefix=False)
        self.vocab_size = config.vocab_size
        self.onehot = ops.OneHot().shard(((parallel_config.dp, 1), (), ()))
        self.on_value = Tensor(float(1 - config.label_smoothing), mstype.float32)
        self.off_value = Tensor(config.label_smoothing / float(self.vocab_size - 1), mstype.float32)
        self.reduce_sum = ops.ReduceSum().shard(((parallel_config.dp, 1),))
        self.reduce_mean = ops.ReduceMean().shard(((1,),))
        self.reshape = ops.Reshape()
        self.last_idx = (-1,)
        self.flatten = ops.Flatten()
        self.neg = ops.Neg().shard(((parallel_config.dp,),))
        self.cast = ops.Cast()
        self.mul = ops.Mul().shard(((parallel_config.dp, 1), (parallel_config.dp, 1)))
        self.mul1 = ops.Mul().shard(((parallel_config.dp,), (parallel_config.dp,)))
        self.sum = ops.ReduceSum().shard(((1,),))
        self.div = ops.RealDiv().shard(((), ()))
        self.eps = 1e-7
        self.add = ops.Add().shard(((), ()))

    # def construct(self, prediction_scores, label_ids, label_weights, seq_length):
    def construct(self, prediction_scores, label_ids, attention_mask):
        """Defines the computation performed."""
         
        attention_mask = attention_mask.astype(mstype.float32)
        label_ids = self.reshape(label_ids, (-1,))
        one_hot_labels = self.onehot(label_ids, self.vocab_size, self.on_value, self.off_value)

        per_example_loss = self.neg(
            self.reduce_sum(self.mul(prediction_scores, one_hot_labels), self.last_idx))  # bs*seq
        mask_loss = self.mul1(per_example_loss, attention_mask.view(-1))
        total_loss = self.sum(mask_loss)
        valid_token = self.sum(attention_mask.view(-1))
        valid_token = self.add(valid_token, self.eps)
        loss = self.div(total_loss, valid_token)
        # loss = self.reduce_mean(per_example_loss)
        return loss
