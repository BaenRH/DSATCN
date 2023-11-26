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
import matplotlib.mlab as plt
import math
import scipy.io as io
from scipy.signal import butter, lfilter

EPS = 1e-12

parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
parser.add_argument('--model_path', type=str, default ='LUCNNfuse32_10_best.pth.tar',    #################################
                    help='Path to model file created by training')
parser.add_argument('--data_dir', type=list, default=['./data/mixdata/foldtrain.txt',
                                                      './data/mixdata/foldtest.txt'],
                    help='directory of fold')
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--batch_size', default=16, type=int,
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
parser.add_argument('--T', default=3, type=int,
                    help='Number of transformer blocks in each repeat')
parser.add_argument('--P', default=11, type=int,
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


def evaluate(args):
    # Load model
    package = torch.load(args.model_path)
    model = TasNet(args.N, args.B, args.sr, args.L, args.X, args.R, args.P, args.C)

    #model.load_state_dict(package['model'])
    key = model.load_state_dict(package['model'], strict=False)
    print(key)
    from thop import profile
    input = torch.randn(1,1024)
    flops, params = profile(model, inputs=(input,input ))
    print(flops, params)

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
    snr_test = 1
    idx = np.where(SNRs_test == snr_test)[0]
    # 不分档则以下2行注释
    # noiseEEG_test = noiseEEG_test[idx]
    # EEG_test = EEG_test[idx]

    evaluate_dataset = EEGData(noiseEEG_test, EEG_test)
    evaluate_loader = DataLoader(evaluate_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

    with torch.no_grad():
        Snr = 1.5
        RRMSE_total = []
        CC_total = []
        SNR = []
        RRMSEspec_total = []
        for i, data in enumerate(evaluate_loader):
            # Get batch data

            eeg_mix = data['eeg_mix'].type(torch.float32)
            # emg = data['emg'].type(torch.float32)
            eeg_clean = data['eeg_clean'].type(torch.float32)
            # n = torch.sqrt(torch.sum(eeg_clean**2,dim=1))/(1*5000)/(Snr*torch.sqrt(torch.sum(emg**2,dim=1))/(1*5000))
            # n = torch.diag(n)
            # eeg_mix = eeg_clean + torch.matmul(n,emg)

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

            estimate_source1, estimate_source2 = model(lpeeg_mix, eeg_mix)
            # estimate_source2  = model(eeg_mix)
            e_noise = eeg_clean - estimate_source2.squeeze()
            snr = 10 * torch.log10(torch.sum(eeg_clean ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + EPS) + EPS)

            estimate2 = estimate_source2.cpu().numpy().squeeze()
            # emg = emg.numpy()
            eeg_clean = eeg_clean.cpu().numpy()
            eeg_mix = eeg_mix.cpu().numpy()

            for j in range(estimate2.shape[0]):
                eeg_snr = snr[j].item()
                SNR.append(eeg_snr)

                e_noise = eeg_clean[j, :] - estimate2[j, :]
                RRMSE = np.sqrt(np.sum(e_noise ** 2) / (np.sum(eeg_clean[j, :] ** 2) + EPS))
                eeg_RRMSE = RRMSE

                Pxx_eeg, _ = plt.psd(eeg_clean[j, :], NFFT=1024)
                Pxx_estimate, _ = plt.psd(estimate2[j, :], NFFT=1024)
                RRMSEspec = np.sqrt(np.sum((Pxx_estimate - Pxx_eeg) ** 2) / (np.sum(Pxx_eeg ** 2) + EPS))

                # 计算CC
                cc = calc_corr(eeg_clean[j, :], estimate2[j, :])
                # print('RRMSE:', eeg_RRMSE)
                # print('CC:', cc)
                RRMSE_total.append(eeg_RRMSE)
                CC_total.append(cc)
                RRMSEspec_total.append(RRMSEspec)

        return RRMSE_total, CC_total, SNR, RRMSEspec_total


if __name__ == '__main__':
    args = parser.parse_args()
    #args.model_path = 'checkpoint/EEGARNet_model/epoch80.pth.tar'
    args.model_path = 'checkpoint/EEGARNet_model/LUCNNfuse32_10_best.pth.tar' ##############################
    #args.model_path = os.path.join('checkpoint/EEGARNet_model',args.model_path)
    print(args)
    RRMSE_total, CC_total, SNR, RRMSEspec_total = evaluate(args)
    meanRRMSEspec = np.round(np.mean(RRMSEspec_total), 4)
    meanRRMSE = np.round(np.mean(RRMSE_total), 4)
    meanSNR = np.round(np.mean(SNR), 4)
    varRRMSE = np.round(np.var(RRMSE_total), 4)
    meanCC = np.round(np.mean(CC_total), 4)
    varCC = np.round(np.var(CC_total), 4)
    io.savemat('result/result_sample.mat', {'RRMSE_total': np.array(RRMSE_total), 'CC_total': np.array(CC_total),
                                            'SNR': SNR, 'RRMSEspec_total': RRMSEspec_total})
    print(meanRRMSEspec)
    print(meanRRMSE)
    print(meanCC)
    print(meanSNR)
