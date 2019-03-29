import torch
import torch.nn as nn
import numpy as np
from dataloader import get_data_loader
from evaluate import evaluate
import torch.optim as optim
import time, os
from model import FCN32s
from evaluate import cross_entropy2d


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir

        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.verbose = args.verbose

        self.model = FCN32s()
        self.parameter = self.model.parameters()
        self.optimizer = optim.Adam(self.parameter, lr=args.learning_rate)

        self.train_dataloader = get_data_loader(self.data_dir, self.batch_size, split='train')
        self.test_dataloader = get_data_loader(self.data_dir, self.batch_size, split='test')

    def train(self):
        self.train_hist = {
            'loss': [],
            'per_epoch_time': [],
            'total_time': []
        }

        print('training start!!')
        start_time = time.time()

        self.model.train()
        best_iou = 0
        for epoch in range(self.epoch):
            self.train_epoch(epoch, self.verbose)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                res = self.evaluate()
                print('Evaluation: Epoch %d: Iou_mean: %.4f, Acc: %.4f, Loss: %.4f,  ' % (
                    epoch + 1, res['iou_mean'], res['acc'], res['loss']))
                print("IOU:", list(res['iou']))
                if res['iou_mean'] > best_iou:
                    self._save_model('best')

        # finish all epoch
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    def train_epoch(self, epoch, verbose=False):
        epoch_start_time = time.time()
        loss_buf = []
        num_batch = int(len(self.train_dataloader.dataset) / self.batch_size)
        for iter, (img, msk) in enumerate(self.train_dataloader):
            if self.gpu_mode:
                img = img.cuda()
                msk = msk.cuda()
            # forward
            self.optimizer.zero_grad()
            output = self.model(img)
            loss = cross_entropy2d(output, msk)
            # backward
            loss.backward()
            self.optimizer.step()
            loss_buf.append(loss.detach().cpu().numpy())

        # finish one epoch
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(np.mean(loss_buf))
        print('Epoch %d: Loss: %.4f, time %.4f s' % (epoch + 1, np.mean(loss_buf), epoch_time))
        # print(f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}, time {epoch_time:.4f}s')

    def evaluate(self):
        self.model.eval()
        res = evaluate(self.model, self.test_dataloader, self.gpu_mode)
        self.model.train()
        return res

    def _save_model(self, epoch):
        torch.save(self.model.state_dict(), self.save_dir + "_" + str(epoch) + '.pkl')
        print("Save model to %s.pkl" % self.save_dir)

    def _load_pretrain(self, epoch):
        self.model.load(self.model.state_dict(), self.save_dir + "_" + str(epoch) + '.pkl')
        print("Load model from %s.pkl" % self.save_dir)
