# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 21:52:05 2021

@author: Mr_gzc
"""

import torch
import torch.nn as nn


class fcNN(nn.Module):
    def __init__(self, lenth):
        super(fcNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(lenth, lenth),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(lenth, lenth),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(lenth, lenth),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(lenth, lenth)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, win):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=win, stride=1, padding=(win - 1) // 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=win, stride=1, padding=(win - 1) // 2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=win, stride=1, padding=(win - 1) // 2),
            nn.BatchNorm1d(32),
            nn.ReLU())

    def forward(self, x):
        res1 = x
        # print(res1.shape)
        out1 = self.conv1(x)
        # print(out1.shape)
        res2 = res1 + out1
        out2 = self.conv1(res2)
        out = res2 + out2
        return out


class ResCNN(nn.Module):
    def __init__(self, len_signal):
        super(ResCNN, self).__init__()
        self.convin = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU())

        self.resblock1 = ResBlock(3)
        self.resblock2 = ResBlock(5)
        self.resblock3 = ResBlock(7)

        self.convout = nn.Sequential(
            nn.Conv1d(96, 32, kernel_size=1, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU())
        self.fc = nn.Linear(32 * ((len_signal - 5) // 1 + 1), len_signal)

    def forward(self, input):
        if input.dim() == 2:
            input = input.unsqueeze(1)  # 二维的数据要扩展成3维，(B, 1, T)
        batch_size = input.size(0)
        out1 = self.convin(input)
        # print(out1.shape)
        blockout1 = self.resblock1(out1)
        blockout2 = self.resblock2(out1)
        blockout3 = self.resblock3(out1)
        out2 = torch.cat([blockout1, blockout2, blockout3], 1)
        # print(out2.shape)
        out3 = self.convout(out2)
        # print(out3.shape)
        o1 = out3.view(batch_size, -1)
        # print(o1.shape)
        out = self.fc(out3.view(batch_size, -1))
        return out


class f_Conv(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=1, kernel_size=3, pool_size=2):
        super(f_Conv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, stride=stride, padding=padding, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, stride=stride, padding=padding, kernel_size=kernel_size),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=pool_size)
        )

    def forward(self, x):
        out = self.conv_block(x)

        return out


class f_Conv_drop(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=1, kernel_size=3, pool_size=2):
        super(f_Conv_drop, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, stride=stride, padding=padding, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, stride=stride, padding=padding, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(kernel_size=pool_size)
        )

    def forward(self, x):
        out = self.conv_block(x)

        return out


class Novel_CNN2(nn.Module):
    def __init__(self, win=3, channel=1, filters=[32, 64, 128, 256, 512, 1024, 2048], len_signal=512):
        super(Novel_CNN2, self).__init__()

        self.conv1 = f_Conv(channel, filters[0], kernel_size=win, padding=(win - 1) // 2)
        self.conv2 = f_Conv(filters[0], filters[1], kernel_size=win, padding=(win - 1) // 2)
        self.conv3 = f_Conv(filters[1], filters[2], kernel_size=win, padding=(win - 1) // 2)
        self.conv4 = f_Conv_drop(filters[2], filters[3], kernel_size=win, padding=(win - 1) // 2)
        self.conv5 = f_Conv_drop(filters[3], filters[4], kernel_size=win, padding=(win - 1) // 2)
        self.conv6 = f_Conv_drop(filters[4], filters[5], kernel_size=win, padding=(win - 1) // 2)
        self.conv7 = nn.Sequential(
            nn.Conv1d(filters[5], filters[6], stride=1, padding=(win - 1) // 2, kernel_size=win),
            nn.ReLU(),
            nn.Conv1d(filters[6], filters[6], stride=1, padding=(win - 1) // 2, kernel_size=win),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(filters[6] * (len_signal // 2 ** 6), len_signal)  # 6是池化的层数

    def forward(self, input):
        if input.dim() == 2:
            input = input.unsqueeze(1)
        layer1out = self.conv1(input)
        layer2out = self.conv2(layer1out)
        layer3out = self.conv3(layer2out)
        layer4out = self.conv4(layer3out)
        layer5out = self.conv5(layer4out)
        layer6out = self.conv6(layer5out)
        layer7out = self.conv7(layer6out)
        flatten1 = self.flatten(layer7out)
        out = self.dense(flatten1)

        return out


class Novel_CNN(nn.Module):
    def __init__(self, len_signal=512, win=3):
        super(Novel_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=win, padding=(win - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=win, padding=(win - 1) // 2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=win, padding=(win - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=win, padding=(win - 1) // 2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=win, padding=(win - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=win, padding=(win - 1) // 2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=win, padding=(win - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=win, padding=(win - 1) // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=win, padding=(win - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=win, padding=(win - 1) // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(kernel_size=2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=win, padding=(win - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, kernel_size=win, padding=(win - 1) // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(kernel_size=2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(1024, 2048, kernel_size=win, padding=(win - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(2048, 2048, kernel_size=win, padding=(win - 1) // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AvgPool1d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8192, len_signal)

    def forward(self, input):
        if input.dim() == 2:
            input = input.unsqueeze(1)
        layer1out = self.conv1(input)
        # print(layer1out.shape)
        layer2out = self.conv2(layer1out)
        # print(layer1out.shape)
        layer3out = self.conv3(layer2out)
        layer4out = self.conv4(layer3out)
        layer5out = self.conv5(layer4out)
        layer6out = self.conv6(layer5out)
        layer7out = self.conv7(layer6out)
        flatten1 = self.flatten(layer7out)
        # print(flatten1.shape)

        out = self.fc(flatten1)

        return out


def testNet():
    x = torch.rand(2, 512)
    nnet = Novel_CNN(512)
    # print(nnet)
    x = nnet(x)
    print(x.shape)
    s1 = x[0]
    # print(s1.shape)


if __name__ == "__main__":
    testNet()
