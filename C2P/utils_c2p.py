# -*- coding: utf-8 -*-
"""
:File: utils_c2p.py
:Author: zhoudl@mail.ustc.edu.cn
"""
import torch


class Callback:
    def __init__(self, optimizer, factor=0.5, patience=10, threshold=1e-4, min_lr=1e-6):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.best = torch.inf
        self.num_bad_epochs = 0

    def step(self, metrics):
        current = float(metrics)
        if current < self.best * (1. - self.threshold):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = float(param_group['lr']) * self.factor
                self.num_bad_epochs = 0
                if max([float(group['lr']) for group in self.optimizer.param_groups]) < self.min_lr:
                    return False
        return True


class Model(torch.nn.Module):
    def __init__(self, output_shape=1, dropout=0.25):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(6, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, output_shape)
        )

    def forward(self, x):
        return self.seq(x)
