import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import GELU


class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(cLN, self).__init__()

        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel * (time_step + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


def repackage_hidden(h):
    """
    Wraps hidden states in new Variables, to detach them from their history.
    """

    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class DepthConv1d(nn.Module):  # 1维残差卷积块
    '''
     # 1-D dilated convolution block
     '''

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        # input_channel：特征维度
        # hidden_channel：隐藏层维度=特征维度*4
        # kernel：3
        # dilation: distance of each kernel
        super(DepthConv1d, self).__init__()

        self.causal = causal
        self.skip = skip

        if self.causal:  # 因果与否，只需要改变padding.(padding是两端补零)
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding

        self.dconv1d1 = nn.Conv1d(input_channel, input_channel, 7, dilation=dilation,
                                  groups=input_channel,
                                  padding='same')
        self.dconv1d2 = nn.Conv1d(input_channel, input_channel, 11, dilation=dilation,
                                  groups=input_channel,
                                  padding='same')
        self.dconv1d3 = nn.Conv1d(input_channel, input_channel, 15, dilation=dilation,
                                  groups=input_channel,
                                  padding='same')
        # self.se = SENet(channels=input_channel*3)

        self.conv1d = nn.Conv1d(3 * input_channel, hidden_channel, 1, padding='same')

        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.dropout = nn.Dropout(0.1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(3 * hidden_channel, eps=1e-08)
            self.reg2 = cLN(3 * hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, 3 * input_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, 3 * hidden_channel, eps=1e-08)

        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        input = self.dropout(input)
        output1 = self.dconv1d1(input)
        output2 = self.dconv1d2(input)
        output3 = self.dconv1d3(input)
        output = torch.cat([output1, output2, output3], 1)

        output = self.reg1(output)

        output = self.nonlinearity2(self.conv1d(output))

        # 分支
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual


# ----------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, dim),
        #     nn.Dropout(dropout)
        # )
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, out_dim, num_head=4, qkv_bias=False, attn_drop=0., proj_drop=0.2):
        super(Attention, self).__init__()

        assert dim % num_head == 0, 'dim should be divisible by num_heads'
        self.num_head = num_head
        head_dim = dim // num_head
        self.scale = head_dim ** (-0.5)

        self.qkv_m = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = nn.LayerNorm(dim)  #

        self.proj_m = nn.Linear(dim, out_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, u_m):
        B, L, dim = u_m.shape

        qkv_m = self.qkv_m(u_m)
        qkv_m = qkv_m.reshape(B, L, 3, self.num_head, dim // self.num_head).permute(2, 0, 3, 1, 4)
        q_m, k_m, v_m = qkv_m.unbind(0)  # torch.Size([4, 8, 1152, 160])  (bs, num_head, length, channel)

        attn_m = (q_m @ k_m.transpose(-2, -1)) * self.scale
        attn_m = attn_m.softmax(dim=-1)
        attn_m = self.attn_drop(attn_m)

        x_m = (attn_m @ v_m).transpose(1, 2).reshape(B, L, dim)
        x_m = self.norm1(x_m)

        x_m = self.proj_m(x_m)
        x_m = self.proj_drop(x_m)

        return x_m


class Transformer(nn.Module):
    def __init__(self, in_channel=64, num_head=4, hidden_channel=256, out_channel=64, dropout=0.1, skip=True):
        super(Transformer, self).__init__()
        self.skip = skip
        self.attn = Attention(dim=in_channel, out_dim=hidden_channel, num_head=num_head, attn_drop=dropout,
                              proj_drop=dropout)
        self.act = nn.GELU()
        self.ff = FeedForward(dim=hidden_channel, hidden_dim=out_channel)
        if self.skip:
            self.skip_ff = FeedForward(dim=hidden_channel, hidden_dim=out_channel)

    def forward(self, x):
        # 在内部转置
        x = x.permute(0, 2, 1)

        x = self.attn(x)  #
        x = self.act(x)
        residual = self.ff(x)
        if self.skip:
            skip = self.skip_ff(x)
            return residual.permute(0, 2, 1), skip.permute(0, 2, 1)


# -------------------------------------------
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# class ConvMod(nn.Module):
#     def __init__(self, dim):
#         super().__init__()

#         self.grn = GRN(256)
#         self.norm = nn.GroupNorm(1, dim, eps=1e-08)
#         self.a = nn.Sequential(
#             nn.Conv1d(dim, 4*dim, 1),
#             nn.GELU(),
#             self.grn,
#             nn.Dropout(0.1),
#             nn.Conv1d(4*dim, dim, 1)
#         )


#         self.v = nn.Conv1d(dim, dim, 1)
#         self.proj = nn.Conv1d(dim, dim, 1)
#         self.skip_ff = nn.Conv1d(dim, dim, 1)

#     def forward(self, x):
#         B, C, H = x.shape
#         x = self.norm(x)
#         x_f, x_res = torch.chunk(x, 2, dim=1)

#         y = torch.fft.rfft(x_f)
#         y_imag = y.imag
#         y_real = y.real
#         y_f = torch.cat([y_real, y_imag], dim=1)
#         y = self.a(y_f)

#         y_real, y_imag = torch.chunk(y, 2, dim=1)
#         y = torch.complex(y_real,y_imag)
#         y = torch.fft.irfft(y, n=H)

#         y = torch.cat([y, x_res], dim=1)
#         x = y * self.v(x)
#         residual = self.proj(x)
#         skip = self.skip_ff(x)

#         return residual, skip

class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.grn = GRN(4 * dim)
        self.norm = nn.GroupNorm(1, dim, eps=1e-08)
        self.a = nn.Sequential(
            nn.Conv1d(2 * dim, 4 * dim, 1),
            nn.GELU(),
            self.grn,
            nn.Dropout(0.1),
            nn.Conv1d(4 * dim, 2 * dim, 1)
        )

        self.v = nn.Conv1d(dim, dim, 1)
        self.proj = nn.Conv1d(dim, dim, 1)
        self.skip_ff = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        B, C, H = x.shape
        x = self.norm(x)

        y = torch.fft.rfft(x)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=1)
        y = self.a(y_f)

        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft(y, n=H)

        x = y * self.v(x)
        residual = self.proj(x)
        skip = self.skip_ff(x)
        # print(residual.size())
        # print(skip.size())

        return residual, skip


class ConvFuse(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.grn = GRN(4 * dim)
        self.norm = nn.GroupNorm(1, dim, eps=1e-08)
        self.a = nn.Sequential(
            nn.Conv1d(2 * dim, 4 * dim, 1),
            nn.GELU(),
            self.grn,
            nn.Dropout(0.1),
            nn.Conv1d(4 * dim, 1 * dim, 1)  ####
        )

        self.v = nn.Conv1d(dim, dim // 2, 1)  #####
        self.proj = nn.Conv1d(dim // 2, dim // 2, 1)

    def forward(self, x):
        B, C, H = x.shape
        x = self.norm(x)

        y = torch.fft.rfft(x)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=1)
        y = self.a(y_f)

        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft(y, n=H)

        # print(y.size())
        # print((self.v(x)).size())
        x = y * self.v(x)
        residual = self.proj(x)
        return residual


#################
class CCFuse(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.v = nn.Sequential(
            nn.GELU(),  ################
            nn.Conv1d(in_channels=dim // 2, out_channels=dim // 2, kernel_size=3, padding='same', bias=True),
            nn.GELU()
        )

        self.p = nn.Conv1d(dim, dim // 2, 1)

    def forward(self, x):
        B, C, T = x.shape
        # x = self.norm(x)
        y = torch.chunk(x, 2, 1)
        y1 = y[1]
        y2 = y[0]
        y2 = self.v(y2)
        m1 = y1 - torch.unsqueeze(y1.mean(dim=-1), dim=-1)
        y1d = torch.sqrt(m1.pow(2).sum(dim=-1))
        m2 = y2 - torch.unsqueeze(y2.mean(dim=-1), dim=-1)
        y2d = torch.sqrt(m2.pow(2).sum(dim=-1))
        q = (m1 * m2).sum(dim=-1) / (y1d * y2d + 1e-08)  # B,C,1
        a = torch.sigmoid(q)  ######################
        a = torch.unsqueeze(a, dim=-1)  # B,C,1
        z = torch.cat((y2 * a, y1), dim=1)
        # output = self.p(z) + y2                    #######################
        output = self.p(z)
        return output

        # print(y.size())
        # print((self.v(x)).size())


class CTRNFuse(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.GroupNorm(1, dim, eps=1e-08)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=3, padding='same', groups=dim)
        self.depthwise_half = nn.Conv1d(dim // 2, dim // 2, kernel_size=3, padding='same', groups=dim // 2)
        self.pointwise = nn.Conv1d(dim, dim // 2, kernel_size=1)
        self.f = nn.Sequential(
            self.depthwise,
            self.norm,
            self.pointwise,
            nn.GELU()
        )
        self.gamma = nn.Parameter(torch.zeros(1, dim // 2, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim // 2, 1))
        self.p = nn.Conv1d(dim // 2, dim // 2, 1)

    def forward(self, x):
        B, C, T = x.shape
        y = self.f(x)

        yt = self.p(y)
        Gt = torch.norm(yt, p=2, dim=(1), keepdim=True)
        Nt = Gt / (Gt.mean(dim=-2, keepdim=True) + 1e-6)

        yc = self.depthwise_half(y)
        Gc = torch.norm(yc, p=2, dim=(2), keepdim=True)
        Nc = Gc / (Gc.mean(dim=-1, keepdim=True) + 1e-6)

        z = self.gamma * (y * Nt * Nc) + self.beta + y
        return self.p(z)


class AFF(nn.Module):
    def __init__(self, dim, r=2):
        super(AFF, self).__init__()
        self.local_att = nn.Sequential(
            nn.Conv1d(dim, dim // r, kernel_size=1),
            #   nn.BatchNorm1d(dim // r),
            nn.GroupNorm(1, dim // r, eps=1e-08),
            nn.ReLU(inplace=True),
            # nn.Conv1d(dim // r, dim, kernel_size=1),
            # #   nn.BatchNorm1d(dim)
            # nn.GroupNorm(1, dim, eps=1e-08)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim // r, kernel_size=1),
            # nn.BatchNorm1d(dim // r),
            nn.GroupNorm(1, dim // r, eps=1e-08),
            nn.ReLU(inplace=True),
            # nn.Conv1d(dim // r, dim, kernel_size=1),
            # # nn.BatchNorm1d(dim)
            # nn.GroupNorm(1, dim, eps=1e-08)
        )

        self.p1 = nn.Sequential(nn.Conv1d(dim // r, dim, kernel_size=1),
            nn.GroupNorm(1, dim, eps=1e-08))
        self.p2 = nn.Sequential(nn.Conv1d(dim // r, dim, kernel_size=1),
            nn.GroupNorm(1, dim, eps=1e-08))

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        xo = x * self.p1(xlg) + residual * self.p2(xlg)
        return xo


# --------------------------------------------------
class myAFF2(nn.Module):
    def __init__(self, dim, r=4):
        super(myAFF, self).__init__()
        self.local_att = nn.Sequential(
            nn.Conv1d(dim * 2, dim * 2 // r, kernel_size=1),
            nn.BatchNorm1d(dim * 2 // r),
            # nn.GroupNorm(1, dim*2 // r, eps=1e-08),
            nn.ReLU(),
            # nn.Dropout(0.05),
            nn.Conv1d(dim * 2 // r, dim, kernel_size=1),
            nn.BatchNorm1d(dim)
            # nn.GroupNorm(1, dim, eps=1e-08)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim * 2, dim * 2 // r, kernel_size=1),
            nn.BatchNorm1d(dim * 2 // r),
            # nn.GroupNorm(1, dim*2 // r, eps=1e-08),
            nn.ReLU(),
            # nn.Dropout(0.05),
            nn.Conv1d(dim * 2 // r, dim, kernel_size=1),
            nn.BatchNorm1d(dim)
            # nn.GroupNorm(1, dim, eps=1e-08)
        )
        self.global_attMax = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Conv1d(dim * 2, dim * 2 // r, kernel_size=1),
            nn.BatchNorm1d(dim * 2 // r),
            nn.ReLU(),
            nn.Conv1d(dim * 2 // r, dim, kernel_size=1),
            nn.BatchNorm1d(dim)
        )
        self.global_temp_att3 = nn.Sequential(
            nn.Conv1d(2, dim, kernel_size=3, padding='same'),  ######
            nn.BatchNorm1d(dim)
            # nn.GroupNorm(1, dim, eps=1e-08)
        )

        self.global_temp_att1 = nn.Sequential(
            nn.Conv1d(2, dim, kernel_size=1),  ######
            nn.BatchNorm1d(dim)
            # nn.GroupNorm(1, dim, eps=1e-08)
        )

        self.local_temp_att = nn.Sequential(
            nn.Conv1d(dim * 2, dim, kernel_size=3, groups=dim, padding='same'),
            nn.BatchNorm1d(dim)
        )
        self.sigmoid = nn.Sigmoid()

        self.weight1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.weight2 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, residual):
        #       xa = x + residual
        xa = torch.cat((x, residual), dim=1)
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        # xg2 = self.global_attMax(xa)
        xatranspose = xa.transpose(1, 2)
        maxpool = nn.AdaptiveMaxPool1d(1)
        avgpool = nn.AdaptiveAvgPool1d(1)
        y1 = maxpool(xatranspose)
        y1 = y1.transpose(1, 2)
        y2 = avgpool(xatranspose)
        y2 = y2.transpose(1, 2)
        y = torch.cat((y1, y2), dim=1)
        yg3 = self.global_temp_att3(y)
        # yg = self.local_temp_att(xa)
        # yg1 = self.global_temp_att1(y)

        xlg = xl + xg + yg3  # + yg1

        # wei = self.sigmoid(xlg)
        # xo = x * wei + residual * (1 - wei)

        w1 = self.weight1(xlg)
        w2 = self.weight2(xlg)
        xo = x * w1 + residual * w2
        return xo


##################################

class myAFF(nn.Module):
    def __init__(self, dim, r=4):
        super(myAFF, self).__init__()
        self.local_att = nn.Sequential(
            nn.Conv1d(dim * 2, dim * 2 // r, kernel_size=1),
            nn.BatchNorm1d(dim * 2 // r),
            nn.ReLU(),
            nn.Conv1d(dim * 2 // r, dim, kernel_size=1),
            nn.BatchNorm1d(dim)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim * 2, dim * 2 // r, kernel_size=1),
            nn.BatchNorm1d(dim * 2 // r),
            nn.ReLU(),
            nn.Conv1d(dim * 2 // r, dim, kernel_size=1),
            nn.BatchNorm1d(dim)
        )
        self.global_attMax = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Conv1d(dim * 2, dim * 2 // r, kernel_size=1),
            nn.BatchNorm1d(dim * 2 // r),
            nn.ReLU(),
            nn.Conv1d(dim * 2 // r, dim, kernel_size=1),
            nn.BatchNorm1d(dim)
        )
        self.global_temp_att3 = nn.Sequential(
            nn.Conv1d(4, dim, kernel_size=3, padding='same'),  ######
            nn.BatchNorm1d(dim)
        )

        self.global_temp_att1 = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=1),  ######
            nn.ReLU(),
            nn.BatchNorm1d(1),
            nn.Conv1d(1, dim, kernel_size=3, padding='same')
        )

        self.local_temp_att = nn.Sequential(
            nn.Conv1d(dim * 2, dim, kernel_size=3, groups=dim, padding='same'),
            nn.BatchNorm1d(dim)
        )
        self.sigmoid = nn.Sigmoid()

        self.weight1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.weight2 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, residual):
        xa = torch.cat((x, residual), dim=1)
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xatranspose = xa.transpose(1, 2)
        # xat1, xat2 = xatranspose.chunk(2, dim=2)
        maxpool = nn.AdaptiveMaxPool1d(1)
        avgpool = nn.AdaptiveAvgPool1d(1)
        y1 = avgpool(xatranspose)
        y1 = y1.transpose(1, 2)
        y2 = maxpool(xatranspose)
        y2 = y2.transpose(1, 2)
        # y3 = avgpool(xat2)
        #  y3 = y3.transpose(1, 2)
        #  y4 = maxpool(xat2)
        #    y = torch.cat((y1, y2, y3, y4), dim=1)
        # yg3 = self.global_temp_att3(y)

        y = torch.cat((y1, y2), dim=1)
        yg1 = self.global_temp_att1(y)

        xlg = xl + xg + yg1

        w1 = self.weight1(xlg)
        w2 = self.weight2(xlg)
        xo = x * w1 + residual * w2
        return xo


##################################

class myAFFshuffle(nn.Module):
    def __init__(self, dim, r=4):
        super(myAFFshuffle, self).__init__()
        self.local_att = nn.Sequential(
            nn.Conv1d(dim * 2, dim * 2 // r, kernel_size=1),
            nn.BatchNorm1d(dim * 2 // r),
            nn.ReLU()
            # nn.Conv1d(dim*2 // r, dim, kernel_size=1),
            # nn.BatchNorm1d(dim)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim * 2, dim * 2 // r, kernel_size=1),
            nn.BatchNorm1d(dim * 2 // r),
            nn.ReLU()
            # nn.Conv1d(dim*2 // r, dim, kernel_size=1),
            # nn.BatchNorm1d(dim)
        )

        self.local_att2 = nn.Sequential(
            nn.Conv1d(dim * 2 // r, dim, kernel_size=1),
            nn.BatchNorm1d(dim)
        )
        self.global_att2 = nn.Sequential(
            nn.Conv1d(dim * 2 // r, dim, kernel_size=1),
            nn.BatchNorm1d(dim)
        )

        self.global_temp_att3 = nn.Sequential(
            nn.Conv1d(2, dim, kernel_size=3, padding='same'),  ######
            nn.BatchNorm1d(dim)
        )

        self.global_temp_att1 = nn.Sequential(
            nn.Conv1d(2, dim, kernel_size=1),  ######
            nn.BatchNorm1d(dim)
        )

        self.sigmoid = nn.Sigmoid()

        self.weight1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.weight2 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, residual):
        #       xa = x + residual
        xa = torch.cat((x, residual), dim=1)

        xlbs = self.local_att(xa)
        xgbs = self.global_att(xa)
        xgbsmatrix = xgbs.repeat(1, 1, 256)
        xlbs1, xlbs2 = xlbs.chunk(2, dim=1)
        xgbs1, xgbs2 = xgbsmatrix.chunk(2, dim=1)
        xl = torch.cat((xlbs1, xgbs1), dim=1)
        xg = torch.cat((xlbs2, xgbs2), dim=1)
        xl = self.local_att2(xl)
        xg = self.global_att2(xg)

        xatranspose = xa.transpose(1, 2)
        maxpool = nn.AdaptiveMaxPool1d(1)
        avgpool = nn.AdaptiveAvgPool1d(1)
        y1 = maxpool(xatranspose)
        y1 = y1.transpose(1, 2)
        y2 = avgpool(xatranspose)
        y2 = y2.transpose(1, 2)
        y = torch.cat((y1, y2), dim=1)
        yg3 = self.global_temp_att3(y)

        xlg = xl + xg + yg3

        w1 = self.weight1(xlg)
        w2 = self.weight2(xlg)
        xo = x * w1 + residual * w2
        return xo


class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim,
                 layer, stack, atten, kernel=3, skip=True,
                 causal=False, dilated=True, ECA=False):
        # 输入维度：编码维度
        # 输出维度：编码维度*声源数
        # BN维度：特征维度
        # 隐藏层维度：特征维度*4
        # 单个TCN层数：8
        # TCN重复数：3
        # 1维卷积残差块的卷积核尺寸：3
        # 是否skip
        # 是否因果
        # 是否采用空洞卷积

        super(TCN, self).__init__()

        # input is a sequence of features of shape (B, N, L)

        # normalization
        if not causal:  # 若因果，则使用累加归一化，否则使用全局归一化
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        self.BN = nn.Conv1d(input_dim, BN_dim, 1)  # 这一层是否能理解为降维？

        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated

        self.stack = stack
        self.layer = layer

        self.TCN = nn.ModuleList([])
        for s in range(stack):  # TCN重复次数
            for i in range(layer):  # TCN的层数
                if self.dilated:  # 是否使用空洞卷积；！！！padding和dilation必须配合好，这样才能保持输入与输出的长度相同
                    self.TCN.append(DepthConv1d(BN_dim, hidden_dim, kernel, dilation=2 ** i, padding=2 ** i, skip=skip,
                                                causal=causal))
                else:
                    self.TCN.append(
                        DepthConv1d(BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal))
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2 ** i
                    else:
                        self.receptive_field += (kernel - 1)
            self.TCN.append(ConvMod(BN_dim))

        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(BN_dim, output_dim, 1)
                                    )
        self.skip = skip
        self.myAFF1 = AFF(64, 4)
        self.myAFF2 = AFF(64, 4)
        self.eca = ECA
        self.se = nn.ModuleList([SENet(hidden_dim // 4, 4) for i in range(layer * stack)])

    def forward(self, input):

        # input shape: (B, N, L)
        skip_connection = ()
        # normalization
        output = self.BN(self.LN(input))

        # pass to TCN
        if self.skip:
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)  # 每个TCN有两个输出
                output = output + residual
                skip_connection = skip_connection + (skip,)

            # self attention
            # output = output.permute(0, 2, 1)
            # output = output + self.attn(output)
            # skip = output + self.ff(output)
            # skip_connection = skip_connection + (skip.permute(0, 2, 1),)

        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        # stage 1
        stage_1 = 0
        for i in range(7):
            stage_1 = stage_1 + skip_connection[i]
        # stage 2
        stage_2 = 0
        for i in range(7, 14):
            stage_2 = stage_2 + skip_connection[i]
        # stage 3
        stage_3 = 0
        for i in range(14, 21):
            stage_3 = stage_3 + skip_connection[i]

        final_mask = self.myAFF1(stage_3, stage_2)
        final_mask = self.myAFF2(final_mask, stage_1)
        return final_mask
        # if self.eca:
        #     final_mask = 0
        #     for connection in skip_connection:
        #         final_mask += connection
        #
        #     se = self.se[0]
        #     # mask_all = 0
        #     # for block_idx in range(self.layer*self.stack):
        #     #     se = self.se[block_idx]
        #     #     attn_final_mask = se(final_mask, skip_connection[block_idx])
        #     #     mask_all += attn_final_mask * skip_connection[block_idx]
        #     attn_final_mask = se(final_mask, final_mask)
        #     return attn_final_mask
        #
        # else:
        #     final_mask = 0
        #     for connection in skip_connection:
        #         final_mask += connection
        #
        #     return final_mask


class SENet(nn.Module):
    def __init__(self, channels, ratio=4):
        super().__init__()
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, all_block, current_block):
        x1 = self.pool(all_block).view(-1, 1, self.channels)
        gates1 = self.conv1(x1)
        gates = self.sigmoid(gates1).view(-1, self.channels, 1)

        x = torch.mul(current_block, gates)

        return x
