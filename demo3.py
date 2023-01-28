"""
garch-midas model
滚动预测
"""

import numpy as np
import pandas as pd
from mfgarch import rolling_predict

pd.set_option("display.max_columns", None)

# 读取数据
df = pd.read_csv("../data/data_mfgarch.csv")

"""
注意: Python的索引是从0开始的, 而R和MATLAB的索引从1开始
窗口1: 上边框索引3253, 下边框位索引7999, 预测索引为8000的波动率
窗口2: 上边框索引3254, 下边框位索引8000, 预测索引为8001的波动率
...
窗口3938: 上边框索引7190, 下边框位索引11936, 预测索引为11937的波动率
"""
result = rolling_predict(data=df,  # 数据框
                         y='return',  # 高频变量
                         x=['dindpro', 'nfci'],  # 双低频变量
                         low_freq=['year_month', 'year_week'],  # 低频变量频率
                         K=[12, 52],  # 低频变量滞后阶数
                         high_start=list(range(3253, 7191)),
                         high_end=list(range(8000, 11938)),
                         gamma=True,  # GJR-GARCH形式: gamma = True; GARCH(1, 1)形式： gamma = False
                         w_restricted=[True, True])  # 权重函数参数w1 = 1: w_restricted = True; w1为待估计参数：w_restricted = False
print(result.predicted_df)  # 参数估计结果
rv = df["rv"].values  # 真实值波动率
rv = rv[8000:11938]
pred = result.predicted_df.iloc[:, 1].values  # 下一天的预测结果
print(np.nanmean((pred - rv) ** 2))  # mse
