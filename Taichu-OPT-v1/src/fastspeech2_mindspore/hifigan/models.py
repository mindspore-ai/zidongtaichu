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
import mindspore
import mindspore.nn as nn
from mindspore import ops
from mindspore.nn import Conv1d, Conv1dTranspose, Conv2d, AvgPool1d
from mindspore import dtype as mstype

LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class AvgPool1d_t(nn.Cell):
    """conv1d"""
    def __init__(self, channels, kernel_size, stride=1, pad_mode="valid", padding=0):
        super(AvgPool1d_t, self).__init__()
        avg_weight = mindspore.ops.Ones()((channels,channels,kernel_size), mindspore.float32)/(channels*channels*kernel_size)
        self.avg_conv = nn.Conv1d(channels, channels, kernel_size, stride, pad_mode,
                                  padding, has_bias=False, weight_init=avg_weight)
        self.avg_conv.weight.requires_grad = False

    def construct(self, x):
        y = self.avg_conv(x)
        return y

class ResBlock(nn.Cell):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.h = h
        self.leaky_relu = nn.LeakyReLU(alpha=LRELU_SLOPE)
        self.convs1 = nn.CellList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                    pad_mode='pad',
                    has_bias=True
                ),
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                    pad_mode='pad',
                    has_bias=True
                ),
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=get_padding(kernel_size, dilation[2]),
                    pad_mode='pad',
                    has_bias=True
                ),
            ]
        )

        self.convs2 = nn.CellList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                    pad_mode='pad',
                    has_bias=True
                ),
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                    pad_mode='pad',
                    has_bias=True
                ),
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                    pad_mode='pad',
                    has_bias=True
                ),
            ]
        )

    def construct(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = self.leaky_relu(x)
            xt = c1(xt)
            xt = self.leaky_relu(xt)
            xt = c2(xt)
            x = xt + x
        return x


class Generator(nn.Cell):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(alpha=LRELU_SLOPE)
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        self.conv_pre = Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3, pad_mode='pad', has_bias=True)

        resblock = ResBlock

        self.ups = nn.CellList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                Conv1dTranspose(
                    h.upsample_initial_channel // (2 ** i),
                    h.upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                    pad_mode='pad',
                    has_bias=True
                )
            )

        self.resblocks = nn.CellList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, pad_mode='pad', has_bias=True)

    def construct(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.leaky_relu(x)
            x = self.ups[i](x)
            xs = self.resblocks[i * self.num_kernels](x)
            for j in range(self.num_kernels-1):
                xs += self.resblocks[i * self.num_kernels + j+1](x)
            x = xs / self.num_kernels
        x = self.leaky_relu(x)
        x = self.conv_post(x)
        x = self.tanh(x)
        return x


class DiscriminatorP(nn.Cell):
    def __init__(self, period, kernel_size=5, stride=3):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.convs = nn.CellList([
            Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1),get_padding(5, 1),0,0), pad_mode='pad', has_bias=True),
            Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1),get_padding(5, 1),0,0), pad_mode='pad', has_bias=True),
            Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1),get_padding(5, 1),0,0), pad_mode='pad', has_bias=True),
            Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1),get_padding(5, 1),0,0), pad_mode='pad', has_bias=True),
            Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2,2,0,0), pad_mode='pad', has_bias=True),
        ])
        self.conv_post = Conv2d(1024, 1, (3, 1), 1, padding=(1,1,0,0), pad_mode='pad', has_bias=True)
        self.leaky_relu = nn.LeakyReLU(alpha=LRELU_SLOPE)

    def construct(self, x):
        fmap = ()

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            pad = nn.Pad(paddings=([0, 0], [0, 0], [0, n_pad]), mode='REFLECT')
            x = pad(x)
            t = t + n_pad
        x = x.reshape(b, c, t // self.period, self.period)

        x = self.leaky_relu(self.convs[0](x))
        fmap+=(x,)
        x = self.leaky_relu(self.convs[1](x))
        fmap+=(x,)
        x = self.leaky_relu(self.convs[2](x))
        fmap+=(x,)
        x = self.leaky_relu(self.convs[3](x))
        fmap+=(x,)
        x = self.leaky_relu(self.convs[4](x))
        fmap+=(x,)
        x = self.conv_post(x)
        fmap+=(x,)
        x = x.reshape(x.shape[0], -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Cell):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.CellList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def construct(self, y, y_hat):
        
        y_d_r1, fmap_r1 = self.discriminators[0](y)
        y_d_g1, fmap_g1 = self.discriminators[0](y_hat)
        y_d_r2, fmap_r2 = self.discriminators[1](y)
        y_d_g2, fmap_g2 = self.discriminators[1](y_hat)
        y_d_r3, fmap_r3 = self.discriminators[2](y)
        y_d_g3, fmap_g3 = self.discriminators[2](y_hat)
        y_d_r4, fmap_r4 = self.discriminators[3](y)
        y_d_g4, fmap_g4 = self.discriminators[3](y_hat)
        y_d_r5, fmap_r5 = self.discriminators[4](y)
        y_d_g5, fmap_g5 = self.discriminators[4](y_hat)
        y_d_rs = (y_d_r1,y_d_r2,y_d_r3,y_d_r4,y_d_r5)
        y_d_gs = (y_d_g1,y_d_g2,y_d_g3,y_d_g4,y_d_g5)
        fmap_rs = (fmap_r1,fmap_r2,fmap_r3,fmap_r4,fmap_r5)
        fmap_gs = (fmap_g1,fmap_g2,fmap_g3,fmap_g4,fmap_g5)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Cell):
    def __init__(self):
        super(DiscriminatorS, self).__init__()
        self.convs = nn.CellList([
            Conv1d(1, 128, 15, 1, padding=7, pad_mode='pad', has_bias=True),
            Conv1d(128, 128, 41, 2, group=4, padding=20, pad_mode='pad', has_bias=True),
            Conv1d(128, 256, 41, 2, group=16, padding=20, pad_mode='pad', has_bias=True),
            Conv1d(256, 512, 41, 4, group=16, padding=20, pad_mode='pad', has_bias=True),
            Conv1d(512, 1024, 41, 4, group=16, padding=20, pad_mode='pad', has_bias=True),
            Conv1d(1024, 1024, 41, 1, group=16, padding=20, pad_mode='pad', has_bias=True),
            Conv1d(1024, 1024, 5, 1, padding=2, pad_mode='pad', has_bias=True),
        ])
        self.conv_post = Conv1d(1024, 1, 3, 1, padding=1, pad_mode='pad', has_bias=True)
        self.leaky_relu = nn.LeakyReLU(alpha=LRELU_SLOPE)

    def construct(self, x):
        fmap = ()
        x = self.leaky_relu(self.convs[0](x))
        fmap+=(x,)
        print('conv1d_1 x[0]',x[0])
        x = self.leaky_relu(self.convs[1](x))
        print('conv1d_2 x[0]',x[0])
        fmap+=(x,)
        x = self.leaky_relu(self.convs[2](x))
        fmap+=(x,)
        x = self.leaky_relu(self.convs[3](x))
        fmap+=(x,)
        x = self.leaky_relu(self.convs[4](x))
        fmap+=(x,)
        x = self.leaky_relu(self.convs[5](x))
        fmap+=(x,)
        x = self.leaky_relu(self.convs[6](x))
        fmap+=(x,)
        x = self.conv_post(x)
        fmap+=(x,)
        x = x.reshape(x.shape[0], -1)
        # print(len(fmap))
        return x, fmap


class MultiScaleDiscriminator(nn.Cell):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.CellList([
            DiscriminatorS(),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpool1 = AvgPool1d_t(channels=1, kernel_size=4, stride=2, pad_mode='pad', padding=2)
        self.meanpool2 = AvgPool1d_t(channels=1, kernel_size=4, stride=2, pad_mode='pad', padding=2)

    def construct(self, y, y_hat):
        y_d_r1, fmap_r1 = self.discriminators[0](y)
        y_d_g1, fmap_g1 = self.discriminators[0](y_hat)
        
        y = self.meanpool1(y)
        y_hat = self.meanpool1(y_hat)
        y_d_r2, fmap_r2 = self.discriminators[1](y)
        y_d_g2, fmap_g2 = self.discriminators[1](y_hat)
        
        y = self.meanpool2(y)
        y_hat = self.meanpool2(y_hat)
        y_d_r3, fmap_r3 = self.discriminators[2](y)
        y_d_g3, fmap_g3 = self.discriminators[2](y_hat)
        
        y_d_rs=(y_d_r1,y_d_r2,y_d_r3)
        fmap_rs=(fmap_r1,fmap_r2,fmap_r3)
        y_d_gs=(y_d_g1,y_d_g2,y_d_g3)
        fmap_gs=(fmap_g1,fmap_g2,fmap_g3)
        return y_d_rs,y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    loss_fn = nn.L1Loss(reduction='mean')
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += loss_fn(gl, rl)

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    loss_fn = nn.MSELoss(reduction='mean')
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = loss_fn(dr, ops.ones_like(dr))
        g_loss = loss_fn(dg, ops.zeros_like(dg))
        loss += (r_loss + g_loss)
        # r_losses.append(r_loss)
        # g_losses.append(g_loss)

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    loss_fn = nn.MSELoss(reduction='mean')
    for dg in disc_outputs:
        l = loss_fn(dg, ops.ones_like(dg))
        # gen_losses.append(l)
        loss += l

    return loss, gen_losses

class DiscriminatorWithLossCell(nn.Cell):
    """连接判别器和损失"""
    def __init__(self, gen, mpd, msd):
        super(DiscriminatorWithLossCell, self).__init__(auto_prefix=False)
        self.gen = gen
        self.mpd = mpd
        self.msd = msd

    def construct(self, x, y):
        """构建判别器损失计算结构"""
        y_g_hat = self.gen(x)
        y_g_hat = ops.functional.stop_gradient(y_g_hat)
        print("y_g_hat[0]:", y_g_hat[0])
        # # MPD
        # y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat)
        # loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat)
        # print(y_ds_hat_r[0])
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        loss_disc_all = loss_disc_s
        print('loss_disc_s', loss_disc_s)
        return loss_disc_all

class GeneratorWithLossCell(nn.Cell):
    """连接生成器和损失"""
    def __init__(self, gen, mpd, msd):
        super(GeneratorWithLossCell, self).__init__(auto_prefix=False)
        self.gen = gen
        self.mpd = mpd
        self.msd = msd

    def construct(self, x, y):
        """构建生成器损失计算结构"""
        y_g_hat = self.gen(x)
        
        # y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
        
        # loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        # loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_all = loss_gen_s + loss_fm_s
        # loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f
        print('gen loss:',loss_gen_all)
        return loss_gen_all
