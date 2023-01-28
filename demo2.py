"""
garch-midas model
基于信息准则的最优滞后阶数选择
"""

import numpy as np
import pandas as pd
from mfgarch import optimal_lag_order

pd.set_option("display.max_columns", None)

# 读取数据
df = pd.read_csv("../data/data_mfgarch.csv")

result = optimal_lag_order(data=df,  # 数据框
                           y='return',  # 高频变量
                           x=["dindpro", "nfci"],  # 低频变量
                           low_freq=["year_month", "year_week"],  # 低频变量频率
                           var_lags={
                               "dindpro": list(range(2, 37)),  # 因子dindpro的滞后阶数有35个可选值2~36
                               "nfci": [52 for _ in range(35)],  # 因子nfci的滞后阶数固定为52
                           },  # 低频变量滞后阶数
                           gamma=True,  # GJR-GARCH形式: gamma = True; GARCH(1, 1)形式： gamma = False
                           w_restricted=[True,
                                         True])  # 权重函数参数w1 = 1: w_restricted = True; w1为待估计参数：w_restricted = False


print(result.AIC_opt) # aic = 29125.810632781555, 最优滞后阶数： lag={'dindpro': 13, 'nfci': 52}
print(result.BIC_opt) # aic = 29191.709178375833, 最优滞后阶数： lag={'dindpro': 13, 'nfci': 52}

