import torch
import torch.nn as nn
from torch.nn import init
from utility import models
import torch.nn.functional as F


# import models

# import visdom
# import models, sdr

# Conv-TasNet
class TasNet(nn.Module):
    def __init__(self, enc_dim=256, feature_dim=64, sr=500, win=16, layer=6, stack=3,
                 kernel=3, num_source=1, causal=False):
        # 编码维度：512
        # 特征维度：128
        # sr:采样率
        # win：窗长，单位（毫秒）
        # 单个TCN层数：8
        # TCN重复数：3
        # 1维卷积残差块的卷积核尺寸：3
        # 声源数：2
        # 是否因果
        super(TasNet, self).__init__()

        # hyper parameters
        self.num_source = num_source

        self.enc_dim = enc_dim  # 编码特征的长度
        self.feature_dim = feature_dim

        self.win = int(sr * win / 1000)
        self.stride = self.win // 2

        self.layer = layer
        self.stack = stack
        self.kernel = kernel

        self.causal = causal

        # input encoder
        # self.encoder1 = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.win)
        self.encoder1 = nn.Conv1d(1, self.enc_dim, self.stride, bias=False, stride=self.stride)
        init.xavier_uniform_(self.encoder1.weight, gain=0.1)

        # self.encoder2 = nn.Conv1d(1, self.enc_dim, self.win, bias=False, stride=self.win)
        self.encoder2 = encoder(1, 256)  # 不要困惑，定义在下面，原来想在卷积后加一个transformer，后来关掉了
        
        # init.xavier_uniform_(self.encoder2.weight, gain=0.1)

        self.BN = nn.Conv1d(self.enc_dim * self.num_source * 2, self.enc_dim * self.num_source, 1)
        # 变量名命名不准确。这里就是一个1*1 channel 降维， 2倍维数降为1倍维数  self.num_source=1

        # TCN separator
        # Low pass
        self.TCN1 = models.TCN(self.enc_dim, self.enc_dim * self.num_source, self.feature_dim, self.feature_dim * 4,
                               self.layer, self.stack, self.kernel, causal=self.causal, ECA=False)
        self.output1 = nn.Sequential(nn.PReLU(),
                                     nn.Conv1d(self.feature_dim, self.enc_dim * self.num_source, 1),
                                     )
        # gates
        self.conv_mid = nn.Conv1d(2*self.enc_dim, 2*self.enc_dim, 5, padding='same', groups=2*self.enc_dim)    # depth-wise conv
        self.conv_mid2 = nn.Sequential(nn.Conv1d(2*self.enc_dim, 2*self.enc_dim, 1), nn.GELU())               # point-wise conv
        #2倍维数进，1倍维数出
        self.grn1 = GRN(2*self.enc_dim)
        self.grn2 = GRN(256)

        # mix
        self.TCN2 = models.TCN(self.enc_dim, self.enc_dim * self.num_source, self.feature_dim, self.feature_dim * 4,
                               self.layer, self.stack, self.kernel, causal=self.causal, ECA=False)
        self.output2 = nn.Sequential(nn.PReLU(),
                                     nn.Conv1d(self.feature_dim, self.enc_dim * self.num_source, 1),
                                     nn.Sigmoid()
                                     )                    #  这里需要与self.output1 统一

        self.decoder1 = nn.ConvTranspose1d(self.enc_dim, 1, kernel_size=self.stride, bias=False, stride=self.stride,) # TODO
        self.decoder2 = Decoder(filters=[256, 512, 1024, 2048], win=3)

        self.fuse = models.ConvMod(512)
        self.firstfuse = models.ConvFuse(512)
        self.corrfuse = models.CCFuse(512)
        self.CTRNfuse = models.CTRNFuse(512)
        self.Sigmoid = nn.Sigmoid()
        self.AFF = models.AFF(256, 2)
        self.myAFF = models.myAFF(256, 4)
        self.myAFFshuffle = models.myAFFshuffle(256, 4)

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:  # 输入只能是2或3维
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)  # 二维的数据要扩展成3维，(B, 1, T)
        batch_size = input.size(0)
        nsample = input.size(2)  # 单条样本长度
        rest = self.win - (self.stride + nsample % self.win) % self.win  # 需要padding的长度，这里要求stride==win/2，win必须为偶数。
        if rest > 0:
            pad = torch.zeros(batch_size, 1, rest).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = torch.zeros(batch_size, 1, self.stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)  # 又在两端padding长度为步长的0

        return input, rest

    def forward(self, input1, input2, return_gate=False, return_mask=False):     # forward 定义类的操作
        output1 = input1.unsqueeze(1)      # B,T——》B,1,T
        output2 = input2.unsqueeze(1)

        batch_size = output1.size(0)

        # waveform encoder
        enc_output1 = self.encoder1(output1)  # B, N, L
        
        # generate masks  lp
        masks_pre1 = self.TCN1(enc_output1)  # masks before split by num_source B*C, N, L
        
        mask1 = self.output1(masks_pre1)
        masks1 = torch.sigmoid(mask1.view(batch_size, self.num_source, self.enc_dim, -1))  # B, C, N, L  多了个C维度   “-1”表示总参数量保持不变，最后一个维度=总参数/前三个维度
        masked_output1 = enc_output1.unsqueeze(1) * masks1  # B, C, N, L
        # print(masked_output1.shape)

        enc_output2 = self.encoder2(output2)  # B, N, L

        #hidden_cat = torch.cat((enc_output2, masked_output1.squeeze(1)), dim=1)

        ## 融合方案1
        # hidden_cat = self.conv_mid(hidden_cat)        ###  拼接后 deptwise 卷积
        # enc_hidden = self.BN(hidden_cat)              ###   再1*1      10.62

        ## 融合方案2
        #hidden_cat = self.conv_mid(hidden_cat)        ###  拼接后卷积
        #enc_hidden = self.firstfuse(hidden_cat)       ###  FFT调制     10.63

        ## 融合方案3  10.12
        ###  拼接后拆分（可简化），enc_output2 卷积然后GeLu,与masked_output1 相关系数大的通道加强，然后拼接，在残差相加
        #enc_hidden = self.corrfuse(hidden_cat)

        ## 融合方案4  CTRN
        # enc_hidden = self.CTRNfuse(hidden_cat)

        ## 融合方案6  仅 pointwise conv
        # enc_hidden = self.BN(hidden_cat)

        ## 融合方案7  AFF
        #enc_hidden = self.AFF(enc_output2, masked_output1.squeeze(1))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~

        ##
        enc_hidden = self.AFF(enc_output2, masked_output1.squeeze(1))
        #enc_hidden = self.myAFFshuffle(enc_output2, masked_output1.squeeze(1))

        #####################################
        masks_pre2 = self.TCN2(enc_hidden)  # masks before split by num_source B*C, N, L

        # combine
        masks_pre2 = self.output2(masks_pre2)
        masks2 = masks_pre2.view(batch_size, self.num_source, self.enc_dim, -1)

        if return_mask:
            return masks2
        masked_output2 = enc_hidden.unsqueeze(1) * masks2  # B, C, N, L

        # print(masked_output1.shape)

        # waveform decoder
        masked_output1 = masked_output1.view(batch_size * self.num_source, self.enc_dim, -1)
        output1 = self.decoder1(masked_output1)  # B*C, 1, L
        # output1 = output1[:, :, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L
        output1 = output1.view(batch_size, self.num_source, -1)  # B, C, T，在调用view之前最好使用contiguous

        masked_output2 = masked_output2.view(batch_size * self.num_source, self.enc_dim, -1)
        output2 = self.decoder2(masked_output2)  # B*C, 1, L
        # output2 = output2[:, :, self.stride:-(rest + self.stride)].contiguous()  # B*C, 1, L
        output2 = output2.view(batch_size, self.num_source, -1)

        return output1, output2


class f_Conv_drop(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=1, kernel_size=3, pool_size=2):
        super(f_Conv_drop, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, stride=stride, padding=padding, kernel_size=kernel_size, groups=output_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(kernel_size=pool_size)
        )

    def forward(self, x):
        out = self.conv_block(x)

        return out


class GRN(nn.Module):
    """ 
    GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    

class encoder(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=1, kernel_size=3, pool_size=2):
        super(encoder, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, stride=4, bias=False, kernel_size=4),
        )
        self.tf = Transformer(in_channel=256, num_head=4, hidden_channel=256, out_channel=256)  

    def forward(self, x):
        out = self.conv_block(x)
        # out = self.tf(out)
        return out


class Decoder(nn.Module):
    def __init__(self, filters=[256, 512, 1024, 2048], win=3):
        super(Decoder, self).__init__()
        self.conv1 = f_Conv_drop(filters[0], filters[1], kernel_size=win, padding=(win - 1) // 2)    #filters[0], filters[0]
        # self.conv2 = f_Conv_drop(filters[1], filters[2], kernel_size=win, padding=(win - 1) // 2)
        # self.conv3 = nn.Sequential(
        #     nn.Conv1d(filters[1], filters[1], stride=1, padding=(win - 1) // 2, kernel_size=win),  #filters[0], filters[1]  8.4621
        #     nn.ReLU(),
        #     nn.Dropout(0.5)    ###################################        0.5
        # )
        self.conv3 = nn.Sequential(
            nn.Conv1d(512,512,7,padding='same',groups=512),
            nn.GELU(),
            nn.Conv1d(512,512,1),
            nn.Dropout(0.5)
        )
        self.tf = nn.Sequential(
            nn.Conv1d(512,512,7,padding='same',groups=512),
            nn.GELU(),
            nn.Conv1d(512,512,1)
        )
        self.flatten = nn.Flatten()
        nn.Dropout(0.5)        ###################################         0
        # self.dense = nn.Linear(512*128, 1024)  # 6是池化的层数
        self.t_linear = nn.Linear(128,8)
        self.c_linear = nn.Linear(256,8)
        self.head = nn.Linear(32832,1024)

    def forward(self, input):
        layer1out = self.conv1(input)
        # layer2out = self.conv2(layer1out)
        layer3out = self.conv3(layer1out) + layer1out
        layer3out = self.tf(layer3out) + layer3out

        x2 =  layer3out[:,:256,:]
        x1 =  layer3out[:,256:,:]

        x1 = self.t_linear(x1).transpose(1,2)
        x1 = self.c_linear(x1).transpose(1,2)
        x1 = torch.flatten(x1,1,2)
        x2 = torch.flatten(x2,1,2)
        # print(x2.shape)
        # print(x1.shape)
        out = self.head(torch.cat((x1,x2),dim=1))

        # flatten1 = self.flatten(layer3out)
        # out = self.dense(flatten1)
        return out


def test_conv_tasnet():
    x = torch.rand(2, 1024)
    nnet = TasNet()
    # print(nnet)
    x, x1 = nnet(x, x)
    print(x.shape)
    s1 = x[0]
    # print(s1.shape)

    
class SENet(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.sequeeze = nn.Linear(in_features=in_channels, out_features=in_channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=in_channels // ratio, out_features=out_channels, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, all_block):
        x1 = self.pool(all_block).view(-1, 1, self.in_channels)
        gates1 = self.relu(self.sequeeze(x1))
        gates1 = self.excitation(gates1)

        gates = self.sigmoid(gates1).view(-1, self.out_channels, 1)

        return gates


## encoder和decoder用到的 Transformer
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
    def __init__(self, in_channel=64, num_head=8, hidden_channel=256, out_channel=64, dropout=0.1, skip=True):
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
        x = x + self.act(self.attn(x))
        x = x +  self.ff(x)
        return x.permute(0, 2, 1)


if __name__ == "__main__":
    test_conv_tasnet()
