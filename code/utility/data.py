# 定义数据类
import scipy.io as scio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt


class EEGData(Dataset):
    def __init__(self, mixed_eeg, clean_eeg):    #函数定义
        self.x_data = mixed_eeg
        self.y_label = clean_eeg

    def butter_lowpass(self, data, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, data)
        return y

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):   # 实际操作
        eeg_mix = self.x_data[index]  # 混合信号
        eeg_clean = self.y_label[index]  # 干净信号（标签）
        lpeeg_mix = self.butter_lowpass(eeg_mix, 20, 512)  # 混合EEG 低通
        lpeeg_clean = self.butter_lowpass(eeg_clean, 20, 512)  # 干净EEG  低通

        sample = {'eeg_mix': eeg_mix, 'eeg_clean': eeg_clean,
                  'lpeeg_mix': lpeeg_mix, 'lpeeg_clean': lpeeg_clean}  # 组成pytorch需要的数据格式
        return sample                     #保存字典，返回


if __name__ == "__main__":
    f_train = np.load('../data/traindata/train_eeg.npz')
    noiseEEG_train, EEG_train = f_train['noiseEEG_train_end_standard'], f_train['EEG_train_end_standard']
    dataset = EEGData(noiseEEG_train, EEG_train)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=1)
    for i, data in enumerate(data_loader):
        if i == 0:
            eeg_mix = data['eeg_mix'].type(torch.float32).numpy()
            eeg_clean = data['eeg_clean'].type(torch.float32).numpy()
            lpeeg_mix = data['lpeeg_mix'].type(torch.float32).numpy()
            lpeeg_clean = data['lpeeg_clean'].type(torch.float32).numpy()

            plt.plot(eeg_mix.squeeze(), 'b', label='eeg_mix')
            plt.legend(loc='upper right', fontsize=13)
            plt.show()

            plt.plot(eeg_clean.squeeze(), 'k', label='eeg_clean')
            plt.legend(loc='upper right', fontsize=13)
            plt.show()

            plt.plot(lpeeg_mix.squeeze(), 'b', label='lpeeg_mix')
            plt.legend(loc='upper right', fontsize=13)
            plt.show()

            plt.plot(lpeeg_clean.squeeze(), 'k', label='lpeeg_clean')
            plt.legend(loc='upper right', fontsize=13)
            plt.show()

            print(i)
            # print(eeg_mix.size())
            # print(eeg_clean.size())
            break
