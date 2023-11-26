import torch
import matplotlib.pyplot as plt
import time
import os
from utility.criterion import cal_loss


# 定义训练类
class Solver(object):
    def __init__(self, data, model, optimizer, scheduler, args):
        self.args = args
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.load_state_dict(package['model'])
            self.optimizer.load_state_dict(package['optimizer'])
            self.start_epoch = int(package['epoch'])
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")  # 赋值正无穷
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0

    def train(self):
        train_loss = []
        test_loss = []
        test_ap_loss = []
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout，训练模式
            start = time.time()
            tr_avg_loss, _, _ = self._run_one_epoch(epoch)
            train_loss.append(tr_avg_loss)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f} '.format(
                epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout，测试模式
            val_loss, lp_loss, al_loss = self._run_one_epoch(epoch, cross_valid=True)
            test_loss.append(al_loss)

            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f} | Valid EEG Loss {3:.3f}'.format(
                epoch + 1, time.time() - start, val_loss, al_loss))
            print('-' * 85)

            # Save model each epoch
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss

            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save({
                    'model': self.model.state_dict(),
                }, file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Adjust learning rate (halving)
            if self.half_lr:  # 学习率衰减
                if val_loss >= self.prev_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv >= 3:  # 验证集loss连续3个epoch不下降
                        self.halving = True
                    if self.val_no_impv >= 10 and self.early_stop:  # loss连续10个epoch不下降
                        print("No imporvement for 10 epochs, early stopping.")
                        break  # 跳出epoch循环，结束训练阶段
                else:
                    self.val_no_impv = 0
            if self.halving:
                optim_state = self.optimizer.state_dict()  # 优化器参数
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0  # 学习率减半
                self.optimizer.load_state_dict(optim_state)  # 更新衰减后的学习率
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            self.prev_val_loss = val_loss
            print('Current learning rate:', self.optimizer.state_dict()['param_groups'][0]['lr'])

            # Save the best model
            if al_loss < self.best_val_loss:
                self.best_val_loss = al_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save({'model': self.model.state_dict(), 'eeg_loss': test_loss,
                            },
                           file_path)  # 保存最好的模型
                print("Find better validated model, saving to %s" % file_path)
            # if epoch == 80:
            #     break

        # 绘制loss和acc================================================================
        plt.figure()
        plt.plot(train_loss, 'g', linewidth=3.0, label='Training Loss')
        plt.plot(test_loss, 'y', linewidth=3.0, label='Validation Loss')
        plt.legend(fontsize=18)
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('Loss Curves', fontsize=16)
        plt.grid(True)
        plt.savefig("./losscurve.model.png")

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0
        lp_loss = 0
        ap_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, data in enumerate(data_loader):
            eeg_mix = data['eeg_mix'][:, 0:5000].type(torch.float32)
            eeg_clean = data['eeg_clean'].type(torch.float32).unsqueeze(1)
            lpeeg_mix = data['lpeeg_mix'].type(torch.float32)
            lpeeg_clean = data['lpeeg_clean'].type(torch.float32).unsqueeze(1)

            if self.use_cuda:
                eeg_mix = eeg_mix.cuda()  # 变量传入cuda必须要对其赋值，
                lpeeg_mix = lpeeg_mix.cuda()
                eeg_clean = eeg_clean.cuda()
                lpeeg_clean = lpeeg_clean.cuda()

            if not cross_valid:  # 如果是验证集就不需要反向传播了
                output1, output2 = self.model(lpeeg_mix, eeg_mix)
                loss1 = cal_loss(lpeeg_clean, output1)
                loss2 = cal_loss(eeg_clean, output2)
                loss = self.args.ratio * loss1 + loss2

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
            else:
                with torch.no_grad():
                    output1, output2 = self.model(lpeeg_mix, eeg_mix)
                    loss1 = cal_loss(lpeeg_clean, output1)
                    loss2 = cal_loss(eeg_clean, output2)
                    loss = self.args.ratio * loss1 + loss2

            total_loss += loss.item()
            lp_loss += loss1.item()
            ap_loss += loss2.item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                    epoch + 1, i + 1, total_loss / (i + 1),
                    loss.item(), 1000 * (time.time() - start) / (i + 1)),
                    flush=True)

        return total_loss / (i + 1), lp_loss / (i + 1), ap_loss / (i + 1)

