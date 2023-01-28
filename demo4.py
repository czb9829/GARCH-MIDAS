"""
garch-midas model
参数估计及样本外预测
"""
import numpy as np
import pandas as pd
from mfgarch import fit_mfgarch, plot_mfgarch

pd.set_option("display.max_columns", None)

# 读取数据
df = pd.read_csv("../data/data_mfgarch.csv")

# *************************************************************************************
#                             单因子garch-midas模型估计及样本外预测
# *************************************************************************************

# 估计模型
# 后100期为样本外
result = fit_mfgarch(data=df,  # 数据框
                     y='return',  # 高频变量
                     x='nfci',  # 低频变量
                     low_freq='year_week',  # 低频变量频率
                     K=52,  # 低频变量滞后阶数
                     gamma=True,  # GJR-GARCH形式: gamma = True; GARCH(1, 1)形式： gamma = False
                     w_restricted=True,  # 权重函数参数w1 = 1: w_restricted = True; w1为待估计参数：w_restricted = False
                     out_of_sample=100)

print(result.predicted)  # 样本外预测结果
rv = df["rv"].values  # 真实值波动率
rv = rv[-100:]
pred = result.predicted["vol"].values  # 波动率预测值
print(np.nanmean((pred - rv) ** 2))  # mse
