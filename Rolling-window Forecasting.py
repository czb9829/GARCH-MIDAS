"""
garch-midas model
rolling-window forecasting
"""

import numpy as np
import pandas as pd
from mfgarch import rolling_predict

pd.set_option("display.max_columns", None)

# read data
df = pd.read_csv("../data/data_mfgarch.csv")

"""
Note: Python's index starts at 0, while R and MATLAB's index starts at 1
Window 1: using sample data indexed from 3253 to 7999 to predict the volatility of the period indexed 8000
Window 2: using sample data indexed from 3254 to 8000 to predict the volatility of the period indexed 8001
...
Window 3938: using sample data indexed from 7190 to 11936 to predict the volatility of the period indexed 11937
"""
result = rolling_predict(data=df, 
                         y='return', 
                         x=['dindpro', 'nfci'],
                         low_freq=['year_month', 'year_week'], 
                         K=[12, 52], 
                         high_start=list(range(3253, 7191)),
                         high_end=list(range(8000, 11938)),
                         gamma=True, 
                         w_restricted=[True, True]) 
print(result.predicted_df)
rv = df["rv"].values  # real volatility (we use intraday rv as a substitute here)
rv = rv[8000:11938]
pred = result.predicted_df.iloc[:, 1].values  # prediction of the next day's volatility
print(np.nanmean((pred - rv) ** 2))  # mse
