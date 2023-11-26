import torch
from itertools import permutations

EPS = 1e-12     #

# def cal_

def cal_loss( source, estimate_source):
    """ 
    Args:
        source: 信号标签
        estimate_source: 估计信号
     """
    eeg_RRMSE = cal_RRMSE(source,estimate_source)
    loss = torch.mean(eeg_RRMSE) #平均batch_size内每个样本的loss

    return loss #, qrs_snr, f_snr #, max_snr, estimate_source, reorder_estimate_source


def cal_RRMSE(source, estimate_source):
    """ 计算相对均方根误差RRMSE
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()

    e_noise = source - estimate_source
    # RRMSE = torch.sum(e_noise ** 2, dim=2) / (torch.sum(source** 2, dim=2) + EPS) + EPS
    RRMSE = torch.sum(abs(e_noise), dim=2) / (torch.sum(abs(source), dim=2) + EPS)
    eeg_RRMSE = RRMSE 

    return eeg_RRMSE#, qrs_snr, f_snr


if __name__ == "__main__":
    torch.manual_seed(123)
    B, C, T = 1, 2, 2
    # fake data
    source = torch.randint(4, (B, C, T)).type(torch.float32)
    estimate_source = torch.randint(4, (B, C, T)).type(torch.float32)

    print('source', source)
    print('estimate_source', estimate_source)

    loss = cal_loss(source, estimate_source)
    print('loss', loss)
    # print('max_snr', max_snr)
    # print('reorder_estimate_source', reorder_estimate_source)
