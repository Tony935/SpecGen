# -*- coding: utf-8 -*-
"""
:File: s2p_benchmark.py
:Author: zhoudl@mail.ustc.edu.cn
"""
import sys

import joblib
import numpy as np
import pandas as pd
from pycaret.regression import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils_s2p import *

sys.path.append('../SpecGen')

from utils import *


def get_corr(y_true, y_pred, **kwargs):
    return np.corrcoef(y_true, y_pred)[0, 1]


data_x = pd.read_excel('D:/work/2024_03/outer/data/data.xlsx', sheet_name='UV')
data_y = pd.read_excel('D:/work/2024_03/outer/data/data.xlsx', sheet_name='overpotential')
seed = 0
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=seed)
norm = StandardScaler().fit(y_train)
y_train_ = pd.DataFrame(norm.transform(y_train), index=y_train.index, columns=data_y.columns)
y_test_ = pd.DataFrame(norm.transform(y_test), index=y_test.index, columns=data_y.columns)

setup(
    data=pd.concat([x_train, y_train_], axis=1),
    target='overpotential',
    test_data=pd.concat([x_test, y_test_], axis=1),
    session_id=0
)
add_metric('pearson', 'Pearson', get_corr)
best_models = compare_models(cross_validation=False, sort='Pearson', n_select=100)
preds = [norm.inverse_transform(model.predict(x_test).reshape(-1, 1)).ravel() for model in best_models]

cnn_path = '../SpecGen/model/S2P_Model'
cnn = torch.load(f'{cnn_path}/model.pth')
norm_cnn = joblib.load(f'{cnn_path}/norm.pkl')
lstm_path = 'model/S2P_LSTM'
lstm = torch.load(f'{lstm_path}/model.pth')
norm_lstm = joblib.load(f'{lstm_path}/norm.pkl')
transformer_path = 'model/S2P_Transformer'
transformer = torch.load(f'{transformer_path}/model.pth')
norm_transformer = joblib.load(f'{transformer_path}/norm.pkl')
cnn.eval()
lstm.eval()
transformer.eval()
with torch.no_grad():
    preds.append(norm_cnn.inverse_transform(cnn(torch.Tensor(x_test.values[:, None]).cuda()).cpu().numpy()).ravel())
    preds.append(norm_lstm.inverse_transform(lstm(torch.Tensor(x_test.values[..., None]).cuda()).cpu().numpy()).ravel())
    preds.append(norm_transformer.inverse_transform(transformer(torch.Tensor(x_test.values[:, None]).cuda()
                                                                ).cpu().numpy()).ravel())

index = [*pull()['Model'], 'Convolutional Neural Network', 'Long Short-Term Memory', 'Transformer']
corr = [np.corrcoef(y_test.values.ravel(), pred)[0, 1] for pred in preds]
mae = [mean_absolute_error(y_test.values.ravel() * 1000, pred * 1000) for pred in preds]
rmse = [np.sqrt(mean_squared_error(y_test.values.ravel() * 1000, pred * 1000)) for pred in preds]
r2 = [r2_score(y_test.values.ravel(), pred) for pred in preds]

pd.DataFrame(np.array([corr, mae, rmse, r2]).T, columns=['r', 'MAE', 'RMSE', 'R2'], index=index
             ).sort_values('r', ascending=False).to_excel('s2p_benchmark.xlsx')
