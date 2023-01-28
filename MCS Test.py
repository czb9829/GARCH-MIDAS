from arch.bootstrap import MCS
import numpy as np
import pandas as pd


pd.set_option("display.max_columns", None)

# read data
df = pd.read_csv(r"--------LossFunc.csv") # the csv file contains the values of loss functions
df = df.copy()[:60]

mcs = MCS(df, size=0.1) # threshold of p-value
a = mcs.compute()

# show the p-value of each model
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

