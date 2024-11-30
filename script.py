# %% [markdown]
# # 1. Preprocessing

# %%
import numpy as np
import pandas as pd

# %%
from data_preprocessing import data_preprocessing

account_booking_train, external_parties_train = data_preprocessing(
    "account_booking_train.csv", "external_parties_train.csv"
)
account_booking_test, external_parties_test = data_preprocessing(
    "account_booking_test.csv", "external_parties_test.csv"
)

# %% [markdown]
# # 2. Feature Engineering

# %%
external_parties_train.head()

# %%
import feature_engineering

# Apply feature engineering to training data
external_parties_train = feature_engineering.perform_feature_engineering(
    external_parties_train
)

# Apply feature engineering to test data
external_parties_test = feature_engineering.perform_feature_engineering(
    external_parties_test
)

# %%
external_parties_train.head(15).to_csv("example.csv", index=False)


