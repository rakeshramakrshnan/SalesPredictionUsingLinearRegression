import pandas as pd
import numpy as np
from ggplot import *

# Step1: EXPLORATORY DATA ANALYSIS

# Read the csv files:
test = pd.read_csv("Source/Test_Data.csv")
train = pd.read_csv("Source/Train_Data.csv")

# combine test and train to perform feature engineering
train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index=True, sort=False)
print("Generate the Data Frame")
print(train.shape, test.shape, data.shape)
print("\n")

# See your data sample
data_sample = data.sample(5)
print("The data sample is as follows")
print(data_sample)
print("\n")

# check for columns with missing values
data_check = data.apply(lambda x: sum(x.isnull()))
print("Check for missing values")
print(data_check)
print("\n")

# look at data statistics
data_statistics = data.describe()
print("The description of Data")
print(data_statistics)
print("\n")

# Catagorical unique data
data_unique = data.apply(lambda x: len(x.unique()))
print("The unique values in data are as follows")
print(data_unique)
print("\n")

