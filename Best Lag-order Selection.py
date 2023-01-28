"""
garch-midas model
selection of best lag-order based on AIC and BIC
"""

import numpy as np
import pandas as pd
from mfgarch import optimal_lag_order

pd.set_option("display.max_columns", None)

# read data
df = pd.read_csv("../data/data_mfgarch.csv")

result = optimal_lag_order(data=df, 
                           y='return',  # high-frequency variable
                           x=["dindpro", "nfci"],  # low-frequency variable
                           low_freq=["year_month", "year_week"],
                           var_lags={
                               "dindpro": list(range(2, 37)),  # the lag-order of "dindpro" could be set from 2 to 36
                               "nfci": [52 for _ in range(35)],  # the lag-order of "nfci" is fixed as 52
                           },  
                           gamma=True,  
                           w_restricted=[True,
                                         True]) 


print(result.AIC_opt) # aic = 29125.810632781555, best lag-order： lag={'dindpro': 13, 'nfci': 52}
print(result.BIC_opt) # aic = 29191.709178375833, best lag-order： lag={'dindpro': 13, 'nfci': 52}

