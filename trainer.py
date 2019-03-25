import torch
import torch.nn as nn
import numpy as np
from dataloader import get_data_loader
from evaluate import evaluate
import torch.optim as optim
import time, os


class Trainer(nn.object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.data_dir = args.data_dir

        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode

        self.model = Model()
        self.parameter = self.model.get_parameter()
        self.optimizer = optim.Adam(self.parameter, lr=args.learning_rate)

        self.train_dataloader = get_data_loader(self.data_dir, split='train')
        self.test_dataloader = get_data_loader(self.data_dir, split='test')

    def train(self):
        self.train_hist = {
            'loss': [],
            'per_epoch_time': [],
            'total_time': []
        }

        print('training start!!')
        start_time = time.time()

        self.model.train()
        for epoch in range(self.epoch):
            self.train_epoch(epoch, self.verbose)

            if (epoch + 1) % 5 == 0:
                self.evaluate()

            if (epoch + 1) % 10 == 0:
                self._snapshot(epoch + 1)

        # finish all epoch
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    def evaluate(self):
        self.model.eval()
        evaluate(self.model, self.test_dataloader)
        self.model.train()
        return

    def _snapshot(self, epoch):
        torch.save(self.model.state_dict(), self.save_dir + str(epoch) + '.pkl')
        print(f"Load model to {save_dir + str(epoch)}.pkl")

    def _load_pretrain(self, epoch):
        self.model.load(self.model.state_dict(), self.save_dir + str(epoch) + '.pkl')
        print(f"Load model from {save_dir + str(epoch)}.pkl")
