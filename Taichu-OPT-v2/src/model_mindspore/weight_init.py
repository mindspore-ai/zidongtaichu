"""Weight init utilities."""

import math
import numpy as np
from mindspore.common.tensor import Tensor


def _average_units(shape):
    """
    Average shape dim.
    """
    if not shape:
        return 1.
    if len(shape) == 1:
        return float(shape[0])
    if len(shape) == 2:
        return float(shape[0] + shape[1]) / 2.
    raise RuntimeError("not support shape.")


def weight_variable(shape):
    scale_shape = shape
    avg_units = _average_units(scale_shape)
    scale = 1.0 / max(1., avg_units)
    limit = math.sqrt(3.0 * scale)
    values = np.random.uniform(-limit, limit, shape).astype(np.float32)
    return Tensor(values)


def one_weight(shape):
    ones = np.ones(shape).astype(np.float32)
    return Tensor(ones)


def zero_weight(shape):
    zeros = np.zeros(shape).astype(np.float32)
    return Tensor(zeros)


def normal_weight(shape, num_units):
    norm = np.random.normal(0.0, num_units ** -0.5, shape).astype(np.float32)
    return Tensor(norm)
