"""
garch-midas model
coefficients estimation
"""
import numpy as np
import pandas as pd
from mfgarch import fit_mfgarch, plot_mfgarch

pd.set_option("display.max_columns", None)

# read data
df = pd.read_csv("../data/data_mfgarch.csv")



# *************************************************************************************
#                             single-variable garch-midas model
# *************************************************************************************

# coefficients estimation
result = fit_mfgarch(data=df,  # select the in-sample data
                     y='return',  # high-frequency variable
                     x='nfci',  # low-frequency variable
                     low_freq='year_week',  
                     K=52,  # the lag-order of low-frequency variable
                     gamma=True,  # GJR-GARCH for short-term volatility: gamma = True; GARCH(1, 1)： gamma = False
                     w_restricted=True)  # parameters for the weighting scheme. 
                                         #If set w1 = 1: w_restricted = True; if w1 not restricted：w_restricted = False

print(result.estimate)  # coefficients estimation results
print(result.fitted.head())  # fitting value of short-term and long-term volatility
print(result.llh, result.aic, result.bic)  # log-likeihood function, AIC and BIC
print(result.variance_ratio)  # print the variance_ratio, which measures the contribution of the long-term volatility to the total volatility
print(result.est_weight)  # estimated parameters for the weighting scheme
plot_mfgarch(result)  # visulization of volatility (in the form of variance)
plot_mfgarch(result, is_sqrt=True)  # visulization of volatility (in the form of standard deviation)

# *************************************************************************************
#                             two-variable garch-midas model
# *************************************************************************************
result = fit_mfgarch(data=df,
                     y='return',
                     x=['dindpro', 'nfci'],
                     low_freq=['year_month', 'year_week'],
                     K=[12, 52],
                     gamma=True,
                     theta_restricted=[False, True],
                     w_restricted=[True, True]) 
print(result.estimate)

# visulization
plot_mfgarch(result, is_sqrt=True)

# *************************************************************************************
#                            multi-variable garch-midas model
# *************************************************************************************

result = fit_mfgarch(data=df,
                     y='return',
                     x=['dindpro', 'nai', 'nfci'],
                     low_freq=['year_month', 'year_month', 'year_week'],
                     K=[12, 12, 52], 
                     gamma=True, 
                     w_restricted=[True,
                                   True,
                                   True])  
print(result.estimate)

plot_mfgarch(result, is_sqrt=True)

# *************************************************************************************
#                        garch-midas model with dummy variable I
# *************************************************************************************
year = pd.to_datetime(df["date"]).dt.year
year = np.array(year)
dummy = np.zeros(year.size)
dummy[year > 2000] = 1
df["dummy"] = dummy

# coefficients estimation 
result = fit_mfgarch(data=df,
                     y='return',
                     x='nfci',
                     low_freq='year_week',
                     dummy="dummy",  # including dummy variable into the intercept
                     K=52,
                     gamma=True,  
                     w_restricted=True)
print(result.estimate)

# *************************************************************************************
#                        garch-midas model with dummy variable II
# *************************************************************************************
year = pd.to_datetime(df["date"]).dt.year
year = np.array(year)
dummy = np.zeros(year.size)
dummy[year > 2000] = 1
df["dummy"] = dummy

result = fit_mfgarch(data=df,
                     y='return',
                     x='nfci',
                     low_freq='year_week',
                     dummy="dummy", 
                     interact_term={"x": "nfci", "dummy": "dummy"},  # including a interact term of the dummy variable and a long-term predictor
                     K=52,
                     gamma=True,
                     w_restricted=True)
print(result.estimate) 

# *************************************************************************************
#                        garch-midas model with dummy variable III
# *************************************************************************************
year = pd.to_datetime(df["date"]).dt.year
year = np.array(year)
dummy = np.zeros(year.size)
dummy[year > 2000] = 1
df["dummy"] = dummy

result = fit_mfgarch(data=df, 
                     y='return', 
                     x=['dindpro', 'nfci'],
                     low_freq=['year_month', 'year_week'], 
                     dummy="dummy",
                     interact_term={"x": ["dindpro", "nfci"], "dummy": ["dummy", "dummy"]},
                     K=[12, 52],
                     gamma=True,  
                     w_restricted=[True, True]) 
print(result.estimate)
