#!/usr/bin/env python

'''
对测试集数据进行测试，统计所有数据平均的RRMSE,SNR和CC值
'''

import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utility.data import EEGData
from utility.conv_tasnet_v1 import TasNet
from utility.network import ResCNN, Novel_CNN2, Novel_CNN, fcNN
import matplotlib.pyplot as plt
import math
import scipy.io as io
from scipy.signal import butter, lfilter

EPS = 1e-8

parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
parser.add_argument('--model_path', type=str,
                    help='Path to model file created by training')
parser.add_argument('--data_dir', type=list, default=['./data/mixdata/foldtrain.txt',
                                                      './data/mixdata/foldtest.txt'],
                    help='directory of fold')
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--batch_size', default=1024, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Num_workers')

# Network architecture
parser.add_argument('--N', default=256, type=int,
                    help='Encode dim')
parser.add_argument('--B', default=64, type=int,
                    help='Feature dim')
parser.add_argument('--sr', default=512, type=int,
                    help='Sample rate')
parser.add_argument('--L', default=16, type=int,
                    help='Length of the filters in samples (16=16ms at 1kHZ)')
parser.add_argument('--X', default=6, type=int,
                    help='Number of convolutional blocks in each repeat')
parser.add_argument('--R', default=3, type=int,
                    help='Number of repeats')
parser.add_argument('--P', default=3, type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--C', default=1, type=int,
                    help='Number of speakers')


# 计算相关系数
def calc_corr(a, b):
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)
    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
    # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
    sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
    corr_factor = cov_ab / sq
    return corr_factor


def butter_lowpass(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


def evaluate(args, snr_test=1):
    # Load model
    package = torch.load(args.model_path)
    model = TasNet(args.N, args.B, args.sr, args.L, args.X, args.R, args.P, args.C)
    # model = fcNN(lenth=1024)
    # model = Novel_CNN2(len_signal=1024)
    # model = ResCNN(1024)
    model.load_state_dict(package['model'])
    # print(model)
    # for name in model.state_dict():
    #     print(name)
    # print('encoder1:',model.state_dict()['encoder1.weight'])
    model.eval()
    if args.use_cuda:
        model.cuda()

    # Load data
    f_test = np.load('../eegdenoisenet/testdata512/test_eeg.npz')
    noiseEEG_test, EEG_test, SNRs_test = f_test['noiseEEG_test'], f_test['EEG_test'], f_test['SNRs_test']

    # 选择 信噪比
    idx = np.where(SNRs_test == snr_test)[0]
    # 不分档则以下2行注释
    noiseEEG_test = noiseEEG_test[idx]
    EEG_test = EEG_test[idx]

    evaluate_dataset = EEGData(noiseEEG_test, EEG_test)
    evaluate_loader = DataLoader(evaluate_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

    with torch.no_grad():
        Snr = 1.5

        for i, data in enumerate(evaluate_loader):
            # Get batch data

            eeg_mix = data['eeg_mix'].type(torch.float32)
            # emg = data['emg'].type(torch.float32)
            eeg_clean = data['eeg_clean'].type(torch.float32)

            lpeeg_mix = butter_lowpass(eeg_mix, 20, 500)
            # hpeeg_mix = butter_highpass(eeg_mix,100,500)

            eeg_mix = eeg_mix.type(torch.float32)

            lpeeg_mix = torch.from_numpy(lpeeg_mix).type(torch.float32)
            # hpeeg_mix = torch.from_numpy(hpeeg_mix).type(torch.float32)
            # lpeeg_clean = data['lpeeg_clean']

            # Forward
            if args.use_cuda:
                eeg_clean = eeg_clean.cuda()
                lpeeg_mix = lpeeg_mix.cuda()
                eeg_mix = eeg_mix.cuda()

            estimate_source1, estimate_source2, gate1, gate2 = model(lpeeg_mix, eeg_mix, return_gate=True)

        # 样本数只有560  直接一个batch读完
        gate1 = gate1.squeeze(2).cpu().numpy()
        gate2 = gate2.squeeze(2).cpu().numpy()
        # plt.psd
        sava_path = 'result'
        io.savemat(os.path.join(sava_path,f'gate_{snr}.mat'),{'gate1':gate1, 'gate2':gate2})

        return gate1.sum()/560, gate1.sum(1).std(), gate2.sum()/560, gate2.sum(1).std() #/np.sqrt(560*64)


if __name__ == '__main__':
    args = parser.parse_args()
    args.model_path = 'checkpoint/EEGARNet_model/5664_10_best.pth.tar'
    out_dir = 'result'
    print(args)

    snr_test = -3
    for snr in range(-7,3):
        gate1_mean, gate1_var, gate2_mean,gate2_var = evaluate(args, snr_test=snr)
        print('*'*7, snr, '*'*7)
        print(gate1_mean, gate1_var)
        print(gate2_mean, gate2_var)




