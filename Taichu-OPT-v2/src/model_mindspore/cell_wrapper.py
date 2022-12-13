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
Cell Wrapper For the Parallel Training.
This is an experimental interface that is subject to change and/or deletion.
"""
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.nn.wrap.loss_scale import TrainOneStepWithLossScaleCell, shard_grad_scale, grad_scale
from mindspore.parallel._utils import _get_enable_parallel_optimizer, _get_pipeline_stages
from .utils import _ClipByGlobalNorm
from .parallel_transformer import ParallelConfig as default_transformer_config
import mindspore.nn as nn
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import RowTensor, Tensor
from mindspore.nn.cell import Cell

__all__ = [
    "ParallelTrainOneStepWithLossScaleCell"
]

_grad_scale = C.MultitypeFuncGraph("_grad_scale")
_shard_grad_scale = C.MultitypeFuncGraph("_shard_grad_scale")
_reciprocal = P.Reciprocal()

reciprocal = P.Reciprocal()
clip_grad = C.MultitypeFuncGraph('clip_grad')

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

@clip_grad.register('Number', 'Number', 'Tensor')
def _clip_grad(clip_type, clip_value, grad):
    """Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


@_grad_scale.register('Tensor', 'Tensor')
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register('Tensor', 'RowTensor')
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices, grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)), grad.dense_shape)


_grad_overflow = C.MultitypeFuncGraph('_grad_overflow')
grad_overflow = P.FloatStatus()


@_grad_overflow.register('Tensor')
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)


@_grad_overflow.register('RowTensor')
def _tensor_grad_overflow_row_tensor(grad):
    return grad_overflow(grad.values)


cast = P.Cast()
add_grads = C.MultitypeFuncGraph('add_grads')


@add_grads.register('Tensor', 'Tensor')
def _add_grads(accu_grad, grad):
    return accu_grad + cast(grad, mstype.float32)


update_accu_grads = C.MultitypeFuncGraph('update_accu_grads')


@update_accu_grads.register('Tensor', 'Tensor')
def _update_accu_grads(accu_grad, grad):
    succ = True
    return F.depend(succ, F.assign(accu_grad, cast(grad, mstype.float32)))


accumulate_accu_grads = C.MultitypeFuncGraph('accumulate_accu_grads')


@accumulate_accu_grads.register('Tensor', 'Tensor')
def _accumulate_accu_grads(accu_grad, grad):
    succ = True
    return F.depend(succ, F.assign_add(accu_grad, cast(grad, mstype.float32)))


zeroslike = P.ZerosLike()
reset_accu_grads = C.MultitypeFuncGraph('reset_accu_grads')


@reset_accu_grads.register('Tensor')
def _reset_accu_grads(accu_grad):
    succ = True
    return F.depend(succ, F.assign(accu_grad, zeroslike(accu_grad)))

@_grad_scale.register("Tensor", "Tensor")
def _tensor_grad_scale(scale, grad):
    return grad * _reciprocal(scale)


class ParallelTrainOneStepWithLossScaleCell(TrainOneStepWithLossScaleCell):
    r"""
    Dynamic Loss scale update cell for the parallel training.

    Encapsulation class of global norm for network training. For loss scaling training, the initial loss scaling value
    will be set to be `loss_scale_value`. In each training step, the loss scaling value will be updated by loss
    scaling value/`scale_factor` when there is an overflow. And it will be increased by loss scaling
    value * `scale_factor` if there is no overflow for a continuous `scale_window` steps. This cell is used for Graph
    mode training in which all logic will be executed on device side(Another training mode is normal(non-sink) mode in
    which some logic will be executed on host).

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_sense (Union[Tensor, Cell]): If this value is Cell type, the loss scaling update logic cell.If this value
                                          is Tensor type, Tensor with shape :math:`()` or :math:`(1,)`.
        enable_global_norm (Bool): Use the global norm. Default: True
        clip_norm (int): The clip norm. Default: 1
        parallel_config (ParallelTransformerParallel): the parallel configure. Default: default_transformer_config

    Inputs:
        - **(*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tuple of 3 Tensor, the loss, overflow flag and current loss scaling value.

        - **loss** (Tensor) -  Tensor with shape :math:`()`.
        - **overflow** (Tensor) -  Tensor with shape :math:`()`, type is bool.
        - **loss scaling value** (Tensor) -  Tensor with shape :math:`()`

    Raises:
        TypeError: If dtype of `inputs` or `label` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, Parameter, nn
        >>> from mindspore.ops import operations as P
        >>> from mindspore.nn.wrap.cell_wrapper import WithLossCell
        >>> from mindspore.common import dtype as mstype
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self, in_feature, out_feature):
        ...         super(Net, self).__init__()
        ...         self.weight = Parameter(Tensor(np.ones([in_feature, out_feature]).astype(np.float32)),
        ...                                 name='weight')
        ...         self.matmul = P.MatMul()
        ...
        ...     def construct(self, x):
        ...         output = self.matmul(x, self.weight)
        ...         return output
        ...
        >>> size, in_features, out_features = 16, 16, 10
        >>> #1) when the type of scale_sense is Cell:
        >>> net = Net(in_features, out_features)
        >>> loss = nn.MSELoss()
        >>> optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> net_with_loss = WithLossCell(net, loss)
        >>> manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
        >>> train_network = nn.parallel.ParallelTrainOneStepWithLossScaleCell(net_with_loss, optimizer,
        ...                                                                   scale_sense=manager)
        >>> input = Tensor(np.ones([out_features, in_features]), mindspore.float32)
        >>> labels = Tensor(np.ones([out_features,]), mindspore.float32)
        >>> output = train_network(input, labels)
        >>>
        >>> #2) when the type of scale_sense is Tensor:
        >>> net = Net(in_features, out_features)
        >>> loss = nn.MSELoss()
        >>> optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> net_with_loss = WithLossCell(net, loss)
        >>> inputs = Tensor(np.ones([size, in_features]).astype(np.float32))
        >>> label = Tensor(np.zeros([size, out_features]).astype(np.float32))
        >>> scaling_sens = Tensor(np.full((1), np.finfo(np.float32).max), dtype=mstype.float32)
        >>> train_network = nn.parallel.ParallelTrainOneStepWithLossScaleCell(net_with_loss, optimizer,
        ...                                                                   scale_sense=scaling_sens)
        >>> output = train_network(inputs, label)
    """

    def __init__(self,
                 network,
                 optimizer,
                 scale_sense=None,
                 enable_global_norm=True,
                 clip_norm=1.0,
                 parallel_config=default_transformer_config):
        super(ParallelTrainOneStepWithLossScaleCell,
              self).__init__(network, optimizer, scale_sense)
        if not isinstance(clip_norm, float):
            raise TypeError("clip norm must be a float value.")

        self.network = network
        self.config = parallel_config
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.enable_global_norm = enable_global_norm
        self.clip = None
        self.enabling_pipeline = False
        if enable_global_norm:
            self.clip = _ClipByGlobalNorm(params=self.weights,
                                          clip_norm=clip_norm,
                                          parallel_config=parallel_config)
        if _get_pipeline_stages() > 1:
            self.enabling_pipeline = True
            self.network.add_flags(defer_inline=True)
            self.accu_grads = self.weights.clone(prefix="accu_grads", init="zeros")
            self.micro_size = parallel_config.micro_size
            self.opt_shard = _get_enable_parallel_optimizer()
            self.degree = 1
            self.cast = P.Cast()
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_before_grad = P.NPUClearFloatStatus()
            self.reshape = P.Reshape()

    def construct(self, *args):

        if self.enabling_pipeline:
            res = self._construct_pipeline(*args)
        else:
            res = self._construct_no_pipeline(*args)

        return res

    def _construct_no_pipeline(self, *args):
        """Defines the computation performed for the non-pipeline."""
        weights = self.weights
        # Forward process
        loss = self.network(*args)
        scaling_sens = self.scale_sense

        # alloc status and clear should be right before grad operation
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        # Backward process using loss scale
        grads = self.grad(self.network,
                          weights)(*args,
                                   scaling_sens_filled)

        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(
            F.partial(_grad_scale, scaling_sens), grads)

        if self.enable_global_norm:
            grads = self.clip(grads)

        # Check whether overflow
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # If overflow, surpass weights update
        # if not, update weights
        # if overflow:
        #     succ = False
        # else:
        #     succ = self.optimizer(grads)

        if not overflow:
            loss = F.depend(loss, self.optimizer(grads))

        return loss, cond, scaling_sens, args[-1]

    def _construct_pipeline(self, *args):
        r"""
         Construct function for the pipeline mode
        """
        weights = self.weights
        loss = self.network(*args)
        scaling_sens = self.scale_sense
        init = self.alloc_status()
        status_clear = self.clear_before_grad(init)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*args, scaling_sens_filled)
        init = F.depend(init, grads)
        get_status = self.get_status(init)
        init = F.depend(init, get_status)
        flag_sum = self.reduce_sum(init, (0,))
        loss = F.depend(loss, status_clear)
        if self.opt_shard:
            grads = self.grad_reducer(grads)
            grads = self.hyper_map(F.partial(shard_grad_scale, scaling_sens * self.degree), grads, self.accu_grads)
        else:
            accu_grads = self.grad_reducer(self.accu_grads)
            grads = self.hyper_map(F.partial(grad_scale, scaling_sens * self.degree), grads, accu_grads)

        if self.enable_global_norm:
            grads = self.clip(grads)

        cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if self.loss_scaling_manager is not None:
            overflow = self.loss_scaling_manager(self.scale_sense, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, overflow, scaling_sens, args[-1])
        return F.depend(ret, succ)


class ParallelAccumTrainOneStepWithLossScaleCell(TrainOneStepWithLossScaleCell):
    """Encapsulation class of network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    To mimic higher batch size, gradients are accumulated N times before weight update.

    For distribution mode, allreduce will be implemented after each sub-step and the trailing time
    will be overided by backend optimization pass.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
        accumulation_steps (int): Number of accumulation steps before gradient update. The global batch size =
                                  batch_size * accumulation_steps. Default: 1.
    """

    def __init__(self, network, optimizer, scale_sense,
                 accumulation_steps=1, enable_global_norm=False,
                 parallel_config=default_transformer_config, clip_norm=1.0):
        super(ParallelAccumTrainOneStepWithLossScaleCell, self).__init__(network, optimizer, scale_sense)

        self.network = network
        self.config = parallel_config
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.enable_global_norm = enable_global_norm
        self.clip = None
        self.enabling_pipeline = False
        if enable_global_norm:
            self.clip = _ClipByGlobalNorm(params=self.weights,
                                          clip_norm=clip_norm,
                                          parallel_config=parallel_config)

        self.accumulation_steps = accumulation_steps
        self.one = Tensor([1], mstype.int32)
        self.zero = Tensor([0], mstype.int32)
        self.local_step = Parameter(initializer(0, [1], mstype.int32))
        self.accu_grads = self.weights.clone(prefix='accu_grads', init='zeros')
        self.accu_overflow = Parameter(initializer(0, [1], mstype.int32))
        self.accu_loss = Parameter(initializer(0, [1], mstype.float32))

        self.degree = 1
        self.less_equal = P.LessEqual()
        self.logical_or = P.LogicalOr()
        self.not_equal = P.NotEqual()
        self.select = P.Select()
        self.reshape = P.Reshape()
        self.hyper_map = C.HyperMap()

    def construct(self, *args):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(*args)

        scaling_sens = self.scale_sense

        # update accumulation parameters
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)
        self.local_step = self.select(is_accu_step, self.local_step + self.one, self.one)
        self.accu_loss = self.select(is_accu_step, self.accu_loss + loss, loss)
        mean_loss = self.accu_loss / self.local_step
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)

        # alloc status and clear should be right before gradoperation
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*args, scaling_sens_filled)

        accu_grads = self.hyper_map(add_grads, self.accu_grads, grads)
        scaling = scaling_sens * self.degree * self.accumulation_steps
        grads = self.hyper_map(F.partial(_grad_scale, scaling), accu_grads)
        grads = self.grad_reducer(grads)

        cond = self.get_overflow_status(status, grads)
        overflow = cond
        overflow = self.logical_or(self.not_equal(self.accu_overflow, self.zero), overflow)
        accu_overflow = self.select(overflow, self.one, self.zero)
        self.accu_overflow = self.select(is_accu_step, accu_overflow, self.zero)
        overflow = self.reshape(overflow, (()))

        if is_accu_step:
            succ = False
            accu_succ = self.hyper_map(update_accu_grads, self.accu_grads, accu_grads)
            succ = F.depend(succ, accu_succ)
        else:
            overflow = self.loss_scaling_manager(self.scale_sense, overflow)

            if not overflow:
                if self.enable_global_norm:
                    grads = C.clip_by_global_norm(grads, 1.0, None)
                else:
                    grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)

                succ = self.optimizer(grads)
                accu_succ = self.hyper_map(reset_accu_grads, self.accu_grads)
                succ = F.depend(succ, accu_succ)

        ret = (mean_loss, overflow, scaling_sens, overflow)
        return ret

        # return F.depend(ret, succ)