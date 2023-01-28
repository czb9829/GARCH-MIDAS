"""
garch-midas model
coefficients estimation & out-of-sample forecasts
"""
import numpy as np
import pandas as pd
from mfgarch import fit_mfgarch, plot_mfgarch

pd.set_option("display.max_columns", None)

# read data
df = pd.read_csv("../data/data_mfgarch.csv")

# *************************************************************************************
#             coefficients estimation & out-of-sample forecasts
# *************************************************************************************

# estimation & forecasts
result = fit_mfgarch(data=df,
                     y='return',
                     x='nfci',
                     low_freq='year_week',
                     K=52, 
                     gamma=True, 
                     w_restricted=True, 
                     out_of_sample=100) # using the last 100 observations for out-of-sample forecasts

print(result.predicted) 
rv = df["rv"].values  # real volatility (intraday rv as a substitute)
rv = rv[-100:]
pred = result.predicted["vol"].values  # predicted volatility
print(np.nanmean((pred - rv) ** 2))  # mse
