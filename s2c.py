# -*- coding: utf-8 -*-
"""
:File: s2c.py
:Author: zhoudl@mail.ustc.edu.cn
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import Callback, Model

data_x = pd.read_excel('data/data.xlsx', sheet_name='UV').values[:, np.newaxis, :]
data_y = pd.read_excel('data/data.xlsx', sheet_name='metals')
col = data_y.columns  # ['Co', 'Ni', 'Cu', 'Mg', 'Cd', 'Zn']

file = 'model/S2C_Model'
os.mkdir(file)

for m in col:
    data_y_ = data_y[[m]].values
    seed = 0
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y_, test_size=0.2, random_state=seed)
    norm = StandardScaler().fit(y_train)
    y_train_ = norm.transform(y_train)
    y_test_ = norm.transform(y_test)
    train_data = torch.utils.data.StackDataset(torch.Tensor(x_train).cuda(), torch.Tensor(y_train_).cuda())
    test_data = torch.utils.data.StackDataset(torch.Tensor(x_test).cuda(), torch.Tensor(y_test_).cuda())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True)

    model = Model(data_y_.shape[1], dropout=0.5).cuda()
    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # weight_decay=1e-3
    scheduler = Callback(optimizer, factor=0.5, patience=50, min_lr=1e-6)

    for epoch in range(1000):
        start = time.time()
        model.train()
        train_loss = 0.
        for step, (x, y) in enumerate(train_loader):
            loss = loss_func(model(x), y)
            loss.backward()
            train_loss += loss.item() * x.shape[0]
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        test_loss = 0.
        for x, y in test_loader:
            loss = loss_func(model(x), y)
            test_loss += loss.item() * x.shape[0]

        print(f'Epoch {epoch + 1:04d} | step {step + 1}/{step + 1} | loss {train_loss / len(train_data):.4f}'
              + f' | test loss {test_loss / len(test_data):.4f} | lr {optimizer.param_groups[0]["lr"]:.3e}'
              + f' | time {time.time() - start:.4f}')
        if not scheduler.step(train_loss):
            break

    os.mkdir(f'{file}/{m}')
    torch.save(model, f'{file}/{m}/model.pth')
    joblib.dump(norm, f'{file}/{m}/norm.pkl')
