# -*- coding: utf-8 -*-
"""
:File: utils_s2p.py
:Author: zhoudl@mail.ustc.edu.cn
"""
import torch
from einops import rearrange
from einops.layers.torch import Rearrange


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


class Model_LSTM(torch.nn.Module):
    def __init__(self, hidden_size, lstm_layers, mlp_layers, dropout):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        self.seq = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            *[
                 torch.nn.Linear(hidden_size * 4, hidden_size * 4),
                 torch.nn.ReLU(),
                 torch.nn.Dropout(dropout)
             ] * (mlp_layers - 1),
            torch.nn.Linear(hidden_size * 4, 1)
        )

    def forward(self, x):
        x = self.lstm(x)[0]
        x = torch.cat([x[:, -1, :], x.mean(dim=1)], dim=1)
        return self.seq(x)


"""https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit_1d.py"""


def posemb_sincos_1d(patches, temperature=10000):
    _, n, dim, device = *patches.shape, patches.device
    n = torch.arange(n, device=device)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)
    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim=1)
    return pe


class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = torch.nn.LayerNorm(dim)
        self.attend = torch.nn.Softmax(dim=-1)
        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = torch.nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(torch.nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(torch.nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dim_head, dropout):
        super().__init__()
        assert seq_len % patch_size == 0
        patch_dim = channels * patch_size
        self.to_patch_embedding = torch.nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            torch.nn.LayerNorm(patch_dim),
            torch.nn.Linear(patch_dim, dim),
            torch.nn.LayerNorm(dim),
        )
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.linear_head = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, num_classes)
        )

    def forward(self, series):
        x = self.to_patch_embedding(series)
        pe = posemb_sincos_1d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.linear_head(x)
