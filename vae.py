# -*- coding: utf-8 -*-
"""
:File: vae.py
:Author: zhoudl@mail.ustc.edu.cn
"""
import os
import time

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from utils import Callback, VAE


def loss_function(recon_x, x, mu, logvar, beta):
    BCE = torch.nn.functional.l1_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    return BCE + KLD * beta


data_x = pd.read_excel('data/data.xlsx', sheet_name='UV').values
data_y = pd.read_excel('data/data.xlsx', sheet_name='metals').values
seed = 0
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=seed)
train_data = torch.Tensor(x_train).cuda()
test_data = torch.Tensor(x_test).cuda()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True)

latent_dim = 16
beta = 0.006

model = VAE(data_x.shape[1], latent_dim).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = Callback(optimizer, factor=0.5, patience=50, min_lr=1e-6)

for epoch in range(10000):
    start = time.time()
    model.train()
    train_loss = 0.
    for step, x in enumerate(train_loader):
        recon_x, mu, logvar = model(x)
        loss = loss_function(recon_x, x, mu, logvar, beta)
        loss.backward()
        train_loss += loss.item() * x.shape[0]
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    test_loss = 0.
    for x in test_loader:
        recon_x, mu, logvar = model(x, pooling=False)
        loss = loss_function(recon_x, x, mu, logvar, beta)
        test_loss += loss.item() * x.shape[0]

    print(f'Epoch {epoch + 1:04d} | step {step + 1}/{step + 1} | loss {train_loss / len(train_data):.4f}'
          + f' | test loss {test_loss / len(test_data):.4f} | lr {optimizer.param_groups[0]["lr"]:.3e}'
          + f' | time {time.time() - start:.4f}')
    if not scheduler.step(train_loss):
        break

file = 'model/VAE_Model'
os.mkdir(file)
torch.save(model, f'{file}/model.pth')
