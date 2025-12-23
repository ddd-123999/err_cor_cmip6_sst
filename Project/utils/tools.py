import math

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=7, delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
            self.counter = 0


def adjust_learning_rate(optimizer, epoch, args):

    if args.model == 'SwinUNet_new':
        # cosine衰减阶段
        min_lr = args.learning_rate * 0.1
        progress = (epoch - args.warmup_epochs) / (args.train_epochs - args.warmup_epochs)
        lr = min_lr + 0.5 * (args.learning_rate - min_lr) * (1 + math.cos(math.pi * progress))
    else:
        lr = args.learning_rate * 0.5 ** ((epoch - args.warmup_epochs) // args.step_size)

    # 更新优化器参数
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

