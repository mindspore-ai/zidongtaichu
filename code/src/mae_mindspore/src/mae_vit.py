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
from mindspore import nn
import mindspore.numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import ops as P
from mindspore import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.parallel.nn import TransformerEncoder, TransformerDecoder
from mindspore.common.initializer import initializer, XavierUniform

from .modules import VitStem


class MAEModule(nn.Cell):
    """
    Base Module For MAE.
    """
    def __init__(self, batch_size, image_size, patch_size, masking_ratio=0.75, channels=3):
        super(MAEModule, self).__init__()
        assert 0 < masking_ratio < 1, \
            'masking ratio must be kept between 0 and 1'
        # seq_length
        self.num_patches = (image_size // patch_size) ** 2
        # seq masked number
        self.num_masked = int(masking_ratio * self.num_patches)
        # batch range
        self.batch_range = np.arange(batch_size)[:, None]
        # per patch dim
        self.patch_dim = channels * patch_size ** 2
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        self.rand_indices = P.Sort()(P.UniformReal()((batch_size, self.num_patches)))
        self.masked_indices = self.rand_indices[1][:, :self.num_masked]
        self.unmasked_indices = self.rand_indices[1][:, self.num_masked:]
        self.mask_info = None
        self.encoder = None
        self.decoder = None

    def generate_mask(self):
        self.mask_info = {
            "batch_range": self.batch_range,
            "masked_indices": self.masked_indices,
            "unmasked_indices": self.unmasked_indices,
        }

        return self.mask_info

    def encoder_engine(self):
        """tokens encoder."""
        return self.encoder

    def decoder_engine(self):
        """code decoder."""
        return self.decoder


class MaeEncoder(MAEModule):
    """MAE Encoder, Default is Vit."""
    def __init__(self,
                 batch_size,
                 patch_size,
                 image_size,
                 encoder_layers=12,
                 encoder_num_heads=12,
                 encoder_dim=768,
                 mlp_ratio=4,
                 masking_ratio=0.75,
                 channels=3,
                 initialization=XavierUniform()):
        super(MaeEncoder, self).__init__(batch_size, image_size, patch_size, masking_ratio, channels)

        self.seq_length = self.num_patches - self.num_masked + 1
        self.encoder = TransformerEncoder(batch_size=batch_size, num_layers=encoder_layers,
                                          num_heads=encoder_num_heads, hidden_size=encoder_dim,
                                          ffn_hidden_size=encoder_dim*mlp_ratio,
                                          seq_length=self.seq_length)
        cls_token = Parameter(
            initializer(initialization, (1, 1, encoder_dim)),
            name='cls', requires_grad=True
        )
        self.cls_token = P.Tile()(cls_token, (batch_size, 1, 1))

        self.encoder_pos_embedding = Parameter(
            initializer(initialization, (1, self.num_patches + 1, encoder_dim)),
            name='pos_embedding', requires_grad=True
        )
        self.add = P.Add()
        self.cat = P.Concat(axis=1)
        self.stride_slice = P.StridedSlice()
        self.norm = nn.LayerNorm((encoder_dim,), epsilon=1e-5)
        self.stem = VitStem(encoder_dim, patch_size, image_size)
        self.encoder_input_mask = Tensor(np.ones((batch_size, self.seq_length, self.seq_length)),
                                         mstype.float16)

    def construct(self, img):
        # patch to encoder tokens and add positions
        tokens, patches = self.stem(img)
        # tokens = self.add(tokens, self.encoder_pos_embedding[:, 1:(self.num_patches + 1)])
        encoder_pos_embedding = self.stride_slice(self.encoder_pos_embedding, (0, 1, 0),
                                                  (1, self.encoder_pos_embedding.shape[1],
                                                   self.encoder_pos_embedding.shape[2]), (1, 1, 1))
        tokens = self.add(tokens, encoder_pos_embedding)
        # get the unmasked tokens to be encoded
        tokens = tokens[self.batch_range, self.unmasked_indices]

        # cls_tokens add pos_embedding
        cls_pos_embedding = self.stride_slice(self.encoder_pos_embedding, (0, 0, 0),
                                              (1, 1, self.encoder_pos_embedding.shape[2]),
                                              (1, 1, 1))
        # cls_tokens = self.add(self.cls_token, self.encoder_pos_embedding[:, :1, :])
        cls_tokens = self.add(self.cls_token, cls_pos_embedding)
        
        # concat cls_tokens
        tokens = self.cat((cls_tokens, tokens))

        # attend with vision transformer
        encoded_tokens = self.encoder(tokens, self.encoder_input_mask)[0]
        if self.norm is not None:
            encoded_tokens = self.norm(encoded_tokens)

        return encoded_tokens, patches


class PreTrainMAEVit(MAEModule):
    """Pretrain MAEVit Module."""
    """
    MindSpore MAE.
    """
    def __init__(self,
                 batch_size,
                 patch_size,
                 image_size,
                 encoder_layers=12,
                 decoder_layers=8,
                 encoder_num_heads=12,
                 decoder_num_heads=16,
                 encoder_dim=768,
                 decoder_dim=512,
                 mlp_ratio=4,
                 masking_ratio=0.75,
                 channels=3,
                 initialization=XavierUniform()):
        super(PreTrainMAEVit, self).__init__(batch_size, image_size, patch_size, masking_ratio, channels)
        self.encoder = MaeEncoder(batch_size, patch_size, image_size,
                                  encoder_layers=encoder_layers, encoder_dim=encoder_dim,
                                  encoder_num_heads=encoder_num_heads, mlp_ratio=mlp_ratio,
                                  initialization=initialization)
        # decoder parameters
        self.seq_length = self.encoder.seq_length
        tgt_seq_length = self.num_patches + 1
        self.mask_token = Parameter(P.StandardNormal()((decoder_dim,)))
        self.mask_tokens = P.Tile()(self.mask_token, (batch_size, self.num_masked, 1))
        self.enc_to_dec = nn.Dense(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else P.Identity()
        self.decoder = TransformerDecoder(batch_size=batch_size, num_layers=decoder_layers,
                                          num_heads=decoder_num_heads, hidden_size=decoder_dim,
                                          ffn_hidden_size=decoder_dim * mlp_ratio,
                                          src_seq_length=self.seq_length,
                                          tgt_seq_length=tgt_seq_length)
        self.decoder_pos_embedding = nn.Embedding(tgt_seq_length, decoder_dim)
        self.attention_mask = Tensor(np.ones((batch_size, tgt_seq_length, tgt_seq_length)), mstype.float16)

        self.to_pixels = nn.Dense(decoder_dim, self.patch_dim, has_bias=True)
        self.norm = nn.LayerNorm((decoder_dim,), epsilon=1e-5)

        self.add = P.Add()
        self.cat = P.Concat(axis=1)
        self.mse_loss = nn.MSELoss()
        self.allreduce = P.AllReduce()

    def construct(self, img, label=None):
        # tokens encoder
        encoder_tokens, patches = self.encoder(img)

        # project encoder to decoder dimensions,
        # if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoder_tokens)
        
        # add position embendding for decoder tokens
        img_tokens = decoder_tokens[:, 1:, :]
        cls_tokens = decoder_tokens[:, :1, :]
        decoder_tokens_ = self.add(img_tokens, self.decoder_pos_embedding(self.unmasked_indices))
        decoder_tokens = self.cat((cls_tokens, decoder_tokens_))

        # mask tokens add the positions using the masked indices derived above
        mask_tokens = self.add(self.mask_tokens, self.decoder_pos_embedding(self.masked_indices))

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = self.cat((decoder_tokens, mask_tokens))
        decoded_tokens = self.decoder(decoder_tokens, self.attention_mask)[0]
        
        # normalize decoder tokens
        decoded_tokens = self.norm(decoded_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, self.seq_length:]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # get the patches to be masked for the final reconstruction loss
        masked_patches = patches[self.batch_range, self.masked_indices]

        # calculate reconstruction loss
        recon_loss = self.mse_loss(pred_pixel_values, masked_patches)
        recon_loss = self.allreduce(recon_loss)
        return recon_loss


if __name__ == "__main__":
    context.set_context(mode=0, device_id=7)
    sdn = P.StandardNormal(seed=2)
    image = sdn((2, 3, 256, 256))
    model = MAEVit(batch_size=2, patch_size=16, image_size=256,
                   encoder_layers=8, decoder_layers=8,
                   encoder_num_heads=2, decoder_num_heads=2,
                   encoder_dim=1024, decoder_dim=256,
                   mlp_ratio=4.0, masking_ratio=0.75, pool="cls")
    loss = model(image)
    print("suceesss", loss)
