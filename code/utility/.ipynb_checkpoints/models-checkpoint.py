import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


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


class MultiRNN(nn.Module):
    """
    Container module for multiple stacked RNN layers.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state. The corresponding output should
                    have shape (batch, seq_len, hidden_size).
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, num_layers=1, bidirectional=False):
        super(MultiRNN, self).__init__()

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers, dropout=dropout,
                                         batch_first=True, bidirectional=bidirectional)

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direction = int(bidirectional) + 1

    def forward(self, input):
        hidden = self.init_hidden(input.size(0))
        self.rnn.flatten_parameters()
        return self.rnn(input, hidden)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers * self.num_direction, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_layers * self.num_direction, batch_size, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers * self.num_direction, batch_size, self.hidden_size).zero_())


class FCLayer(nn.Module):
    """
    Container module for a fully-connected layer.

    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, input_size).
        hidden_size: int, dimension of the output. The corresponding output should
                    have shape (batch, hidden_size).
        nonlinearity: string, the nonlinearity applied to the transformation. Default is None.
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=None):
        super(FCLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.FC = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        if nonlinearity:
            self.nonlinearity = getattr(F, nonlinearity)
        else:
            self.nonlinearity = None

        self.init_hidden()

    def forward(self, input):
        if self.nonlinearity is not None:
            return self.nonlinearity(self.FC(input))
        else:
            return self.FC(input)

    def init_hidden(self):
        initrange = 1. / np.sqrt(self.input_size * self.hidden_size)
        self.FC.weight.data.uniform_(-initrange, initrange)
        if self.bias:
            self.FC.bias.data.fill_(0)


class DepthConv1d(nn.Module):  # 1维残差卷积块

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
        
        self.conv1d = nn.Conv1d(3*input_channel, hidden_channel, 1, padding='same')
        
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.dropout = nn.Dropout(0.1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(3*hidden_channel, eps=1e-08)
            self.reg2 = cLN(3*hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, 3*input_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, 3*hidden_channel, eps=1e-08)

        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        input = self.dropout(input)
        output1 = self.dconv1d1(input)
        output2 = self.dconv1d2(input)
        output3 = self.dconv1d3(input)
        output = torch.cat([output1, output2, output3], 1)
        
        output = self.reg1(output)
                        
        output =self.nonlinearity2(self.conv1d(output))
                           
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
        self.norm1 = nn.LayerNorm(dim)  # TODO

        self.proj_m = nn.Linear(dim, out_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, u_m):
        B, L, dim = u_m.shape

        qkv_m = self.qkv_m(u_m)
        qkv_m = qkv_m.reshape(B, L, 3, self.num_head, dim // self.num_head).permute(2, 0, 3, 1, 4)
        q_m, k_m, v_m = qkv_m.unbind(0)  #  torch.Size([4, 8, 1152, 160])  (bs, num_head, length, channel)

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
        self.attn = Attention(dim=in_channel, out_dim=hidden_channel, num_head=num_head, attn_drop=dropout, proj_drop=dropout)
        self.act = nn.GELU()
        self.ff = FeedForward(dim=hidden_channel, hidden_dim=out_channel)
        if self.skip:
            self.skip_ff = FeedForward(dim=hidden_channel, hidden_dim=out_channel)

    def forward(self, x):
        # 在内部转置
        x = x.permute(0, 2, 1)

        x = self.attn(x)  # TODO  这里没有加x
        x = self.act(x)
        residual = self.ff(x)
        if self.skip:
            skip = self.skip_ff(x)
            return residual.permute(0, 2, 1), skip.permute(0, 2, 1)


# --------------------------------------------------
class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim,
                 layer, stack, atten , kernel=3, skip=True,
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
            self.TCN.append(Transformer())
        # for _ in range(3):   # TODO
        #     self.TCN.append(Transformer())

        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(BN_dim, output_dim, 1)
                                    )
        self.skip = skip

        self.eca = ECA
        self.se = nn.ModuleList([SENet(hidden_dim//4, 4) for i in range(layer * stack)])

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

        if self.eca:
            final_mask = 0
            for connection in skip_connection:
                final_mask += connection

            se = self.se[0]
            # mask_all = 0
            # for block_idx in range(self.layer*self.stack):
            #     se = self.se[block_idx]
            #     attn_final_mask = se(final_mask, skip_connection[block_idx])
            #     mask_all += attn_final_mask * skip_connection[block_idx]
            attn_final_mask = se(final_mask, final_mask)
            return attn_final_mask

        else:
            final_mask = 0
            for connection in skip_connection:
                final_mask += connection

            return final_mask







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

