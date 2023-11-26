
# EEGARNETv2c
## $ source activate emo
 ## $ sh run.sh
python train.py  \
--epochs 100  \
--batch_size 32 \
--lr 1e-3  \
--checkpoint 1  \
--model_path 'LUCNNfuse32_10_best.pth.tar'  \
--print_freq 400  \
--ratio 1 \
#&> training.log
python evaluate.py --model_path 'LUCNNfuse32_10_best.pth.tar'


# EEGARNETv2c

#python train.py  \
#--epochs 100  \
#--batch_size 128  \
#--lr 1e-3  \
#--checkpoint 0   \
#--model_path '64_02_best.pth.tar'  \                                                          # should be 128?   otherwise 64 model overlapped
#--print_freq 400  \
#--ratio 0.2

#python evaluate.py --model_path '128_02_best.pth.tar'                                         # error occured! used the overlapped 64 model to evaluate 8.68 dB


# EEGARNETv1
#cd ../EEGARNETv1
#pwd

#python train.py  \
#--epochs 100  \
#--batch_size 128  \
#--lr 1e-3  \
#--checkpoint 0   \
#--model_path '64_10_best.pth.tar'  \                                                          # should be 128?
#--print_freq 400  \
#--ratio 0.2                                                                                   # should be ratio 1?

#python evaluate.py --model_path '64_10_best.pth.tar'                                          # should be 128?
