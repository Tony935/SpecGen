# -*- coding: utf-8 -*-
"""
:File: utils.py
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
    def __init__(self, output_shape=1, dropout=0.5):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv1d(1, 4, 17),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(4),
            torch.nn.MaxPool1d(4),

            torch.nn.Conv1d(4, 16, 17),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(16),
            torch.nn.MaxPool1d(4),

            torch.nn.Conv1d(16, 32, 9),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.MaxPool1d(2),

            torch.nn.Flatten(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, output_shape)
        )

    def forward(self, x):
        return self.seq(x)


class Encoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv1d(1, 4, 17),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(4),
            torch.nn.MaxPool1d(8),
            torch.nn.Flatten(),
        )
        self.mean = torch.nn.Linear(352, latent_dim)
        self.logvar = torch.nn.Linear(352, latent_dim)

    def forward(self, x):
        x = self.seq(torch.unsqueeze(x, 1))
        means = self.mean(x)
        logvars = self.logvar(x)
        return means, logvars


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, output_dim)
        )
        self.p = torch.nn.AvgPool1d(17, stride=1, padding=8, count_include_pad=False)

    def forward(self, x, pooling=False):
        x = self.seq(x)
        if pooling:
            x = self.p(x)
        x[x > 4] = 4
        return x


class VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, pooling=None):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        if pooling is None:
            pooling = not self.training
        return self.decoder(z, pooling), mu, logvar
