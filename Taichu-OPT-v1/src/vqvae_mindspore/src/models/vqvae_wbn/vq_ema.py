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

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter, ms_function
import mindspore.common.dtype as mstype

try:
    from src.models.vqvae_wbn.ema import ExponentialMovingAverage
except:
    from src.vqvae_mindspore.src.models.vqvae_wbn.ema import ExponentialMovingAverage


@ms_function
def get_embedding(a, b):
    return a / b

class VectorQuantizerEMA(nn.Cell):
    """Sonnet module representing the VQ-VAE layer.

          Implements a slightly modified version of the algorithm presented in
          'Neural Discrete Representation Learning' by van den Oord et al.
          https://arxiv.org/abs/1711.00937

          The difference between VectorQuantizerEMA and VectorQuantizer is that
          this module uses exponential moving averages to update the embedding vectors
          instead of an auxiliary loss. This has the advantage that the embedding
          updates are independent of the choice of optimizer (SGD, RMSProp, Adam, K-Fac,
          ...) used for the encoder, decoder and other parts of the architecture. For
          most experiments the EMA version trains faster than the non-EMA version.

          Input any tensor to be quantized. Last dimension will be used as space in
          which to quantize. All other dimensions will be flattened and will be seen
          as different examples to quantize.

          The output tensor will have the same shape as the input.

          For example a tensor with shape [16, 32, 32, 64] will be reshaped into
          [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
          independently.

          Attributes:
            embedding_dim: integer representing the dimensionality of the tensors in the
              quantized space. Inputs to the modules must be in this format as well.
            num_embeddings: integer, the number of vectors in the quantized space.
            commitment_cost: scalar which controls the weighting of the loss terms (see
              equation 4 in the paper).
            decay: float, decay for the moving averages.
            epsilon: small float constant to avoid numerical instability.
          """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost,
                 decay, epsilon=1e-5, dtype=mstype.float32):
        """Initializes a VQ-VAE EMA module.

                   Args:
                     embedding_dim: integer representing the dimensionality of the tensors in
                       the quantized space. Inputs to the modules must be in this format as
                       well.
                     num_embeddings: integer, the number of vectors in the quantized space.
                     commitment_cost: scalar which controls the weighting of the loss terms
                       (see equation 4 in the paper - this variable is Beta).
                     decay: float between 0 and 1, controls the speed of the Exponential Moving
                       Averages.
                     epsilon: small constant to aid numerical stability, default 1e-5.
                     dtype: dtype for the embeddings variable, defaults to tf.float32.
                     name: name of the module.
                   """

        super(VectorQuantizerEMA, self).__init__()

        self.decay = decay
        self.epsilon = epsilon
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        embedding_shape = [embedding_dim, num_embeddings]
        self.embeddings = Parameter(Tensor(np.random.randn(*embedding_shape) * 0.01, dtype), requires_grad=False)

        self.ema_cluster_size = ExponentialMovingAverage(decay, np.zeros(num_embeddings), dtype)
        self.ema_dw = ExponentialMovingAverage(decay, self.embeddings, dtype)

        # define ops
        self.sum = ops.ReduceSum(keep_dims=True)
        self.pow = ops.Pow()
        self.prt = ops.Print()
        self.matmul = ops.MatMul()
        self.argmax = ops.Argmax(output_type=mstype.int32, axis=1)
        self.onehot_on = Tensor(1.0, mstype.float32)
        self.onehot_off = Tensor(0.0, mstype.float32)

    def construct(self, z, is_training):
        """Connects the module to some inputs.

        Args:
          inputs: Tensor, final dimension must be equal to embedding_dim. All other
            leading dimensions will be flattened and treated as a large batch.
          is_training: boolean, whether this connection is to training data. When
            this is set to False, the internal moving average statistics will not be
            updated.

        Returns:
          dict containing the following keys and values:
            quantize: Tensor containing the quantized version of the input.
            loss: Tensor containing the loss to optimize.
            perplexity: Tensor containing the perplexity of the encodings.
            encodings: Tensor containing the discrete encodings, ie which element
            of the quantized space each input element was mapped to.
            encoding_indices: Tensor containing the discrete encoding indices, ie
            which element of the quantized space each input element was mapped to.
        """
        inputs = z.transpose(0, 2, 3, 1)
        ops.stop_gradient(inputs)

        flat_inputs = inputs.reshape((-1, self.embedding_dim))
        distances = (self.sum(flat_inputs**2, 1) -
                     2 * self.matmul(flat_inputs, self.embeddings) +
                     self.sum(self.embeddings**2, 0))

        encoding_indices = self.argmax(-distances)
        encodings = ops.OneHot()(encoding_indices, self.num_embeddings,
                                 self.onehot_on, self.onehot_off).astype(distances.dtype)

        quantized = self.quantize(encodings)
        quantized = quantized.reshape(inputs.shape).transpose(0, 3, 1, 2)
        encoding_indices = encoding_indices.reshape(inputs.shape[:-1])

        loss = (ops.stop_gradient(quantized) - z)**2
        e_latent_loss = loss.mean()

        if is_training:
            dw = ops.matmul(flat_inputs.transpose(1, 0), encodings)
            updated_ema_dw = self.ema_dw.construct(dw)
            updata_ema_cluster_size = self.ema_cluster_size.construct(encodings.sum(axis=0))

            n = ops.ReduceSum(keep_dims=False)(updata_ema_cluster_size)
            updata_ema_cluster_size = ((updata_ema_cluster_size + self.epsilon) /
                                       (n + self.num_embeddings * self.epsilon) * n)

            divb = updata_ema_cluster_size.reshape((1, -1))
            self.embeddings = get_embedding(updated_ema_dw, divb)

        loss = e_latent_loss * self.commitment_cost

        # Straight Through Estimator
        quantized = z + ops.stop_gradient(quantized - z)
        # diff = self.sub(quantized, z)
        # quantized = self.add(z, ops.stop_gradient(diff))
        avg_probs = encodings.mean(axis=0)
        perplexity = ops.Exp()(-(avg_probs * ops.Log()(avg_probs + 1e-10)).sum())

        return {
            'loss': loss,
            'encodings': encodings,
            'distances': distances,
            'quantize' : quantized,
            'perplexity': perplexity,
            'encoding_indices': encoding_indices,
        }

    def quantize_code(self, code):
        # code:     [b, ch, cw]
        # return:   [b, c, ch, cw]
        dtype = self.embeddings.dtype
        encodings = ops.OneHot()(code, self.num_embeddings,
                                 self.onehot_on, self.onehot_off).astype(dtype)
        quantized = self.quantize(encodings)
        quantized = quantized.transpose(0, 3, 1, 2)
        return quantized

    def quantize(self, encodings):
        """Returns embedding tensor for a batch of indices."""
        embeddings = self.embeddings.transpose(1, 0)
        quantized = ops.matmul(encodings, embeddings)
        return quantized

if __name__ == '__main__':
    VQ_EMA = VectorQuantizerEMA(2, 5, 1, 0.9)
    print(f"Initial embed: {VQ_EMA.embeddings}")

    for i in range(3):
        z = Parameter(np.random.randn(4, 2, 8, 8).astype(np.float32), requires_grad=True)
        # print(f"Input z1: {Tensor(z)}")

        out = VQ_EMA.construct(z, True)
        print(f"OUT1 loss: {out['loss']}, ppl: {out['perplexity']}")
        print(f"Updated embed1: {VQ_EMA.embeddings}")

        z = Parameter(np.random.randn(4, 2, 8, 8).astype(np.float32), requires_grad=True)
        # print(f"Input z2: {Tensor(z)}")

        out = VQ_EMA.construct(z, False)
        print(f"OUT2 loss: {out['loss']}, ppl: {out['perplexity']}")
        print(f"Updated embed2: {VQ_EMA.embeddings}")




