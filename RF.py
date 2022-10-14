# %%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# %%
feature_count = pd.read_csv("feature_count_df.csv")
list(feature_count)

# %%
target = feature_count.drop("mLogD7.4")

# %%
RF = RsandomForestRegressor()[],



