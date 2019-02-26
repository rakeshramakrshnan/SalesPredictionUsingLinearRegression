import pandas as pd
import numpy as np
from scipy.stats import mode


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

# The categorical variables are filtered
categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']

# ID cols and source are excluded
categorical_columns = [x for x in categorical_columns if x not in ['Product_Id','Branch_Id','source']]

# The frequency of categories is printed
for col in categorical_columns:
    print ('\nFrequency of Categories for variable %s'%col)
    print (data[col].value_counts())


# Step2: DATA CLEANING

# Determine the average weight per item:
product_avg_weight = data.pivot_table(values='Product_Size', index='Product_Id')

# Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Product_Size'].isnull()

# Impute Product_Size data and check the number of missing values before and after imputation to confirm
print ('Orignal #missing: %d'% sum(miss_bool))
data.loc[miss_bool, 'Product_Size'] = data.loc[miss_bool, 'Product_Id'].apply(lambda x: product_avg_weight.loc[x])
print ('Final #missing: %d'% sum(data['Product_Size'].isnull()))

# Determine the mode for Branch_Size
Branch_Area_mode = data.pivot_table(values='Branch_Area', columns='Branch_Type',aggfunc=(lambda x:mode(x).mode[0]) )
print ('Mode for each Outlet_Type:')
print (Branch_Area_mode)

# Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Branch_Area'].isnull()

# Impute data and check #missing values before and after imputation to confirm
print ('\nOrignal #missing: %d'% sum(miss_bool))
data.loc[miss_bool,'Branch_Area'] = data.loc[miss_bool,'Branch_Type'].apply(lambda x: Branch_Area_mode[x])
print ('Final #missing: %d'%sum(data['Branch_Area'].isnull()))



