import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from transformers import AdamW
import scipy.io as scio
import numpy as np
from utility.conv_tasnet_v1 import TasNet
from utility.network import ResCNN, Novel_CNN2, Novel_CNN, fcNN
from utility.data import EEGData
from utility.solver import Solver
from utility.data_input import prepare_data
import argparse
import os
from torch.optim.lr_scheduler import LambdaLR
import random

# import torchsummary
# =============================================================================
#     def __init__(self, enc_dim=512, feature_dim=128, sr=1000, win=16, layer=4, stack=3, 
#                  kernel=3, num_spk=2, causal=False, visdom_mask=False):
# =============================================================================
parser = argparse.ArgumentParser()
# General config
# Task related   ?????
parser.add_argument('--data_dir', type=list, default=['./data/mixdata/foldtrain.txt', \
                                                      './data/mixdata/foldvalid.txt'],
                    help='directory of fold')
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
parser.add_argument('--P', default=11, type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--C', default=1, type=int,
                    help='Number of speakers')
# parser.add_argument('--ECA', default=1, type=int,
#                    help='是否使用ECA')

# Training config
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=1, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=1, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')

# save and load model   ?????????
parser.add_argument('--save_folder', default='checkpoint/EEGARNet_model/',
                    help='Location to save epoch models')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--model_path', default='best.pth.tar',
                    help='Location to save best validation model')

# batch
parser.add_argument('--shuffle', default=1, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers to generate minibatch')

# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3
                    , type=float,
                    help='Init learning rate')  # TODO:这么小？
parser.add_argument('--ratio', default=1, type=float,)
parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')

# logging
parser.add_argument('--print_freq', default=20, type=int,
                    help='Frequency of printing training information')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom_epoch', dest='visdom_epoch', type=int, default=0,
                    help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom_id', default='TasNet training',
                    help='Identifier for visdom run')


def main(args):
    # train_num = 3000   # how many trails for train
    # test_num = 400     # how many trails for test
    # combin_num = 10    # combin EEG and noise ? times/mnt/DEV/han/eeg/DASTCN_grnFFT
    if not os.path.exists('../../eegdenoisenet/traindata512/train_eeg.npz'):

        EEG_all = np.load('./data/EEG_all_epochs_512hz.npy')
        noise_all = np.load('./data/EMG_all_epochs_512hz.npy')
        noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std_VALUE = prepare_data(
            EEG_all=EEG_all,
            noise_all=noise_all,
            combin_num=10,
            train_per=0.8,
            noise_type='EMG')
    else:
        f_train = np.load('../../eegdenoisenet/traindata512/train_eeg.npz', allow_pickle=True)
        noiseEEG_train, EEG_train = f_train['noiseEEG_train'], f_train['EEG_train']
        f_val = np.load('../../eegdenoisenet/valdata512/val_eeg.npz', allow_pickle=True)
        noiseEEG_val, EEG_val = f_val['noiseEEG_val'], f_val['EEG_val']

    train_dataset = EEGData(noiseEEG_train, EEG_train)    # 不封装成类是无法将数据带入深度学习框架训练的
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)  # 自动将数据转换成tensor

    test_dataset = EEGData(noiseEEG_val, EEG_val)      # 这里是验证集吧
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    data = {'tr_loader': train_loader, 'cv_loader': test_loader}     #继续封装供Solver调用
    # tasnet
    model = TasNet(args.N, args.B, args.sr, args.L, args.X, args.R, args.P, args.C)

    # print(model)
    if args.use_cuda:
        model.cuda()
    # torchsummary.summary(model,input_size =(1,400),device='cpu')
    # torch.save({'model':model.state_dict()},'initmodel.pth.tar')

    # 固定初始化参数
    # package = torch.load(args.model_path)
    # model.load_state_dict(package['model'])
    total_steps = len(train_loader) * args.epochs

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer, scheduler = build_optimizer(args, model, total_steps)

    solver = Solver(data, model, optimizer, scheduler, args)
    solver.train()


def build_optimizer(args, model, train_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    # V1
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=train_steps * 0.1, t_total=train_steps)
    return optimizer, scheduler


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    setup_seed(2024) #2024
    # 确认电脑是否支持CUDA，然后显示CUDA信息
    print(device)
    args = parser.parse_args()
    main(args)
