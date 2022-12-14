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
import mindspore
from mindspore import nn, Tensor
from mindspore import ops as P
from mindspore.common.initializer import initializer, XavierUniform
from src.model_mindspore.parallel_transformer import ParallelConfig

MIN_NUM_PATCHES = 4


class BatchDense(nn.Cell):
    """BatchDense module."""

    def __init__(self, in_features, out_features, initialization, has_bias=True):
        super(BatchDense, self).__init__()
        self.out_features = out_features
        self.dense = nn.Dense(in_features, out_features, has_bias=has_bias)
        self.dense.weight.set_data(initializer(initialization, [out_features, in_features]))
        self.dense.matmul.shard(((ParallelConfig.dp, 1), (1, 1)))
        self.dense.bias_add.shard(((ParallelConfig.dp, 1), (1,)))
        self.dense.weight.parallel_optimizer = False
        self.dense.bias.parallel_optimizer = False

        self.reshape = P.Reshape()
        self.pixel_values = self.dense.weight.shape[-1]

    def construct(self, x):
        bs, seq_len, dim = x.shape
        out = self.reshape(x, (bs * seq_len, dim))
        out = self.dense(out)
        out = self.reshape(out, (bs, seq_len, self.out_features))
        return out


class VitStem(nn.Cell):
    """Stem layer for ViT."""

    def __init__(self, dim, patch_size, image_size, channels=3, initialization=XavierUniform()):
        super(VitStem, self).__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches {num_patches} is too small'
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.patch_to_embedding = BatchDense(patch_dim, dim, initialization, has_bias=True)

    def construct(self, patches):
        # p = self.patch_size
        # bs, channels, h, w = img.shape
        # x = self.reshape(img, (bs, channels, h // p, p, w // p, p))
        # x = self.transpose(x, (0, 2, 4, 1, 3, 5))
        # patches = self.reshape(x, (bs, (h//p)*(w//p), channels*p*p))
        x = self.patch_to_embedding(patches)
        return x, patches


class PatchEmbed(nn.Cell):
    """Construct the embeddings from patch, position embeddings."""
    def __init__(self, img_size=224, patch_size=16, in_features=3, out_features=768):
        super(PatchEmbed, self).__init__()
        self.hybrid = None
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.projection = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                                    kernel_size=patch_size, stride=patch_size,
                                    weight_init="xavier_uniform", has_bias=True)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        # usually not use norm
        # self.norm = _LayerNorm((out_features,), eps=1e-6).to_float(mindspore.float32)

    def construct(self, x):
        """construct"""
        x = self.projection(x)
        x = self.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = self.transpose(x, (0, 2, 1))
        # x = self.norm(x)
        return x

grad_scale = P.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor", "Tensor")
def gradient_scale(scale, grad):
    return grad * P.cast(scale, P.dtype(grad))


class VitTrainOneStepCell(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1.0):
        super(VitTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.hyper_map = P.HyperMap()
        self.reciprocal_sense = Tensor(1 / sens, mindspore.float32)

    def scale_grad(self, gradients):
        gradients = self.hyper_map(P.partial(grad_scale, self.reciprocal_sense), gradients)
        return gradients

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = P.fill(loss.dtype, loss.shape, self.sens)
        # calculate gradients, the sens will equal to the loss_scale
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        # gradients / loss_scale
        grads = self.scale_grad(grads)
        # reduce gradients in distributed scenarios
        grads = self.grad_reducer(grads)
        loss = P.depend(loss, self.optimizer(grads))
        return loss

