## Dual-Stream Attention-TCN for EMG Removal from a Single-Channel EEG

## Introduction

​	Long-term and mobile healthcare applications have increased the use of single-channel electroencephalogram (EEG) systems. However, electromyography (EMG) artifacts often disturb EEGs. The lack of spatial correlation, diversity of waveforms, and time-varying overlap make eliminating EMG interference from a single-channel EEG difficult. To overcome these challenges, we create DSATCN, a dual-stream learning model that makes use of multi-level and multi-scale temporal dependencies in different frequency bands to perform robust EEG reconstruction. The first DSATCN stream extracts low-frequency band EEG features with reduced EMG interference. The second stream selectively combines the high-level features of the first stream with its own low-level features to refine the EEG reconstruction across the entire frequency band, lowering the risk of overfitting. Both streams employ a novel attention-based temporal convolution network (ATCN) to adaptively separate the overlapping features of EEGs and EMGs. The ATCN has multiple stages to represent various temporal dependencies at different levels. Each stage consists of multi-scale dilated convolutions and fast Fourier transform modulations, which efficiently enrich the receptive fields and establish global self-attention mechanisms. The stages' outputs are merged by relaxed attentional feature fusion modules, which bridge semantic gaps between features at various levels. Experimental results show that our DSATCN improves the average signal-to-noise ratio by over 33.82% and reduces the average relative root mean square error by over 21.48%, outperforming the SOTA methods significantly.

## Architecture

![dsatcn](https://github.com/BaenRH/DSATCN/blob/main/photo/dsatcn.png)

## Result

Comparisons of Averaged EEG Reconstruction Accuracies Obtained by Different Methods on the EEGDenoiseNet

|                 | RRMSE_{s}  | RRMSE_{t}  | CC         | SNR (dB)    |
| --------------- | ---------- | ---------- | ---------- | ----------- |
| EEMD-ICA        | 2.4924     | 0.9892     | 0.6963     | 1.3211      |
| SSA-ICA         | 2.1909     | 0.9558     | 0.6981     | 1.8111      |
| 1-D ResCNN      | 0.6329     | 0.6272     | 0.7774     | 4.4702      |
| DeepSeparator   | 0.4611     | 0.6304     | 0.7814     | 4.7404      |
| Novel CNN       | 0.4685     | 0.4496     | 0.8637     | 8.2046      |
| Proposed DSATCN | **0.3726** | **0.3530** | **0.9023** | **10.9798** |



## Installation

```shell
pip install -r requirements.txt
```

## Training

```shell
cd code
sh run.sh	
```

## Evaluate

```shell
cd code
python evalute.py	
```

