from arch.bootstrap import MCS
import numpy as np
import pandas as pd



pd.set_option("display.max_columns", None)

# 读取数据
df = pd.read_csv(r"D:\气候政策不确定性与能源期货价格波动\Manuscript\RMSE120.csv")
df = df.copy()[:60]

mcs = MCS(df, size=0.1)
a = mcs.compute()

# 显示各模型的MSC P值
print("MCS P-values")
print(mcs.pvalues)
# Included: P>0.1
print("Included")
included = mcs.included
print(included)
# Excluded: P<=0.1
print("Excluded")
excluded = mcs.excluded
print(excluded)

