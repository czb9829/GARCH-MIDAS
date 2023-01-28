"""
garch-midas model
参数估计(全样本)
"""
import numpy as np
import pandas as pd
from mfgarch import fit_mfgarch, plot_mfgarch

pd.set_option("display.max_columns", None)

# 读取数据
df = pd.read_csv("../data/data_mfgarch.csv")



# *************************************************************************************
#                             单因子garch-midas模型估计
# *************************************************************************************

# 估计模型
result = fit_mfgarch(data=df,  # 数据框
                     y='return',  # 高频变量
                     x='nfci',  # 低频变量
                     low_freq='year_week',  # 低频变量频率
                     K=52,  # 低频变量滞后阶数
                     gamma=True,  # GJR-GARCH形式: gamma = True; GARCH(1, 1)形式： gamma = False
                     w_restricted=True)  # 权重函数参数w1 = 1: w_restricted = True; w1为待估计参数：w_restricted = False
print(result.estimate)  # 参数估计结果
print(result.fitted.head())  # 长短期波动拟合值
print(result.llh, result.aic, result.bic)  # 对数似然与信息准则
print(result.variance_ratio)  # 低频变量的解释程度
print(result.est_weight)  # 低频变量滞后期的权重
print(result.mf_forecast)  # 波动率预测值, 预测期数由参数n_horizons设定
plot_mfgarch(result)  # 画图: 方差形式的波动率
plot_mfgarch(result, is_sqrt=True)  # 画图: 标准差形式的波动率

# *************************************************************************************
#                             双因子garch-midas模型估计
# *************************************************************************************
result = fit_mfgarch(data=df,  # 数据框
                     y='return',  # 高频变量
                     x=['dindpro', 'nfci'],  # 双低频变量
                     low_freq=['year_month', 'year_week'],  # 低频变量频率
                     K=[12, 52],  # 低频变量滞后阶数
                     gamma=True,  # GJR-GARCH形式: gamma = True; GARCH(1, 1)形式： gamma = False
                     theta_restricted=[False, True],
                     w_restricted=[True, True])  # 权重函数参数w1 = 1: w_restricted = True; w1为待估计参数：w_restricted = False
print(result.estimate)  # 参数估计结果
# 画图
plot_mfgarch(result, is_sqrt=True)

# *************************************************************************************
#                            多因子garch-midas模型估计
# *************************************************************************************

result = fit_mfgarch(data=df,  # 数据框
                     y='return',  # 高频变量
                     x=['dindpro', 'nai', 'nfci'],  # 多低频变量
                     low_freq=['year_month', 'year_month', 'year_week'],  # 低频变量频率
                     K=[12, 12, 52],  # 低频变量滞后阶数
                     gamma=True,  # GJR-GARCH形式: gamma = True; GARCH(1, 1)形式： gamma = False
                     w_restricted=[True,
                                   True,
                                   True])  # 权重函数参数w1 = 1: w_restricted = True; w1为待估计参数：w_restricted = False
print(result.estimate)  # 参数估计结果
# 画图
plot_mfgarch(result, is_sqrt=True)

# *************************************************************************************
#                        包含虚拟变量的garch-midas模型估计1
# *************************************************************************************
year = pd.to_datetime(df["date"]).dt.year
year = np.array(year)
dummy = np.zeros(year.size)
dummy[year > 2000] = 1
df["dummy"] = dummy
# 估计模型
result = fit_mfgarch(data=df,  # 数据框
                     y='return',  # 高频变量
                     x='nfci',  # 低频变量
                     low_freq='year_week',  # 低频变量频率
                     dummy="dummy",  # 截距项中添加虚拟变量dummy
                     K=52,  # 低频变量滞后阶数
                     gamma=True,  # GJR-GARCH形式: gamma = True; GARCH(1, 1)形式： gamma = False
                     w_restricted=True)  # 权重函数参数w1 = 1: w_restricted = True; w1为待估计参数：w_restricted = False
print(result.estimate)  # 参数估计结果

# *************************************************************************************
#                        包含虚拟变量的garch-midas模型估计2
# *************************************************************************************
year = pd.to_datetime(df["date"]).dt.year
year = np.array(year)
dummy = np.zeros(year.size)
dummy[year > 2000] = 1
df["dummy"] = dummy
# 估计模型
result = fit_mfgarch(data=df,  # 数据框
                     y='return',  # 高频变量
                     x='nfci',  # 低频变量
                     low_freq='year_week',  # 低频变量频率
                     dummy="dummy",  # 截距项中添加虚拟变量dummy
                     interact_term={"x": "nfci", "dummy": "dummy"},  # 因子nfci的斜率中添加虚拟变量dummy
                     K=52,  # 低频变量滞后阶数
                     gamma=True,  # GJR-GARCH形式: gamma = True; GARCH(1, 1)形式： gamma = False
                     w_restricted=True)  # 权重函数参数w1 = 1: w_restricted = True; w1为待估计参数：w_restricted = False
print(result.estimate)  # 参数估计结果

# *************************************************************************************
#                        包含虚拟变量的garch-midas模型估计3
# *************************************************************************************
year = pd.to_datetime(df["date"]).dt.year
year = np.array(year)
dummy = np.zeros(year.size)
dummy[year > 2000] = 1
df["dummy"] = dummy
# 估计模型
result = fit_mfgarch(data=df,  # 数据框
                     y='return',  # 高频变量
                     x=['dindpro', 'nfci'],  # 双低频变量
                     low_freq=['year_month', 'year_week'],  # 低频变量频率
                     dummy="dummy",  # 截距项中添加虚拟变量dummy
                     interact_term={"x": ["dindpro", "nfci"], "dummy": ["dummy", "dummy"]},
                     # 因子dindpro的斜率中添加虚拟变量dummy, 因子nfci的斜率中添加虚拟变量dummy
                     K=[12, 52],  # 低频变量滞后阶数
                     gamma=True,  # GJR-GARCH形式: gamma = True; GARCH(1, 1)形式： gamma = False
                     w_restricted=[True, True])  # 权重函数参数w1 = 1: w_restricted = True; w1为待估计参数：w_restricted = False
print(result.estimate)  # 参数估计结果
