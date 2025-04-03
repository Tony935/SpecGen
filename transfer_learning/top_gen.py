# -*- coding: utf-8 -*-
"""
:File: top_gen.py
:Author: zhoudl@mail.ustc.edu.cn
"""
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.append('../SpecGen')

from utils import *

s2p_path = '../SpecGen/model/S2P_Model'
s2p = torch.load(f'{s2p_path}/model.pth')
norm_s2p = joblib.load(f'{s2p_path}/norm.pkl')
vae_path = '../SpecGen/model/VAE_Model'
vae = torch.load(f'{vae_path}/model.pth')

latent_dim = 16
rng = np.random.RandomState(2)

x = torch.Tensor(pd.read_excel('../data/data.xlsx', sheet_name='UV').values).cuda()
data_uv = []
data_op = []
vae.eval()
s2p.eval()
with torch.no_grad():
    y_pred = norm_s2p.inverse_transform(s2p(torch.unsqueeze(x, 1)).cpu().numpy()).ravel()
    mu, logvar = vae.encoder(x[[y_pred.argmin()]])
    for i in range(1000):
        uv = vae.decoder(torch.Tensor(rng.normal(size=(10000, latent_dim))).cuda() * torch.exp(0.5 * logvar) * 3 + mu,
                         pooling=True)
        op = norm_s2p.inverse_transform(s2p(torch.unsqueeze(uv, 1)).cpu().numpy()).ravel()
        data_uv.append(uv[op.argsort()[:20]])
        data_op.append(op[op.argsort()[:20]])
data_uv = torch.cat(data_uv)
data_op = np.concatenate(data_op)
data_uv = data_uv[data_op.argsort()[:20]]

best = {}
for system in ['A', 'B', 'C', 'D']:
    s2c_path = f'model/S2C_Model_{system}'
    pred = []
    col = pd.read_excel(f'../data/transfer_{system}.xlsx', sheet_name='metals').columns
    for m in col:
        s2c = torch.load(f'{s2c_path}/{m}/model.pth')
        norm_s2c = joblib.load(f'{s2c_path}/{m}/norm.pkl')
        s2c.eval()
        with torch.no_grad():
            pred.append(norm_s2c.inverse_transform(s2c(torch.unsqueeze(data_uv, 1)).cpu().numpy()).ravel())
    pred = np.array(pred).T
    pred[pred < 0] = 0
    pred /= pred.sum(axis=1, keepdims=True)
    best[system] = pd.DataFrame(pred, columns=col)
with pd.ExcelWriter(f'best_transfer.xlsx') as writer:
    for system in ['A', 'B', 'C', 'D']:
        best[system].to_excel(writer, sheet_name=system, index=False)
