# -*- coding: utf-8 -*-
"""
:File: grid_search.py
:Author: zhoudl@mail.ustc.edu.cn
"""
import joblib
import numpy as np
import pandas as pd

from utils_c2p import *

c2p_path = 'model/C2P_Model'
c2p = torch.load(f'{c2p_path}/model.pth')
norm_c2p = joblib.load(f'{c2p_path}/norm.pkl')

total = 50
grid = []
for a in range(1, total - 4):
    for b in range(1, total - 3 - a):
        for c in range(1, total - 2 - a - b):
            for d in range(1, total - 1 - a - b - c):
                for e in range(1, total - a - b - c - d):
                    f = total - a - b - c - d - e
                    grid.append([a, b, c, d, e, f])
grid = np.array(grid) / total

c2p.eval()
with torch.no_grad():
    pred = norm_c2p.inverse_transform(c2p(torch.Tensor(grid).cuda()).cpu().numpy()).ravel()

pd.DataFrame(grid[pred.argsort()[:20]], columns=['Co', 'Ni', 'Cu', 'Mg', 'Cd', 'Zn']
             ).to_excel('best.xlsx', sheet_name='best', index=False)
