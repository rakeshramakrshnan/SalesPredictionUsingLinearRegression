import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import mode
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso




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


plt.figure(figsize=(10.7))
sns.heatmap(data.corr())

# Step2: DATA CLEANING

# Determine the average weight per item:
product_avg_weight = data.pivot_table(values='Product_Size', index='Product_Id')

# Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Product_Size'].isnull()

# Impute Product_Size data and check the number of missing values before and after imputation to confirm
print ('Orignal #missing: %d'% sum(miss_bool))
data.loc[miss_bool, 'Prodduct_Size'] = data.loc[miss_bool, 'Product_Id'].apply(lambda x: product_avg_weight.loc[x])
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
data.dropna(subset=['Branch_Area']).pivot_table(values='Branch_Area', columns='Branch_Type',aggfunc=(lambda x:mode(x.astype('str')).mode[0]), dropna=True)
print ('Final #missing: %d'%sum(data['Branch_Area'].isnull()))

# Step 3: MOODEL BUILDING

# Define target and ID columns:
target = 'Branch_Product_Sales'
IDcol = ['Product_Id','Branch_Id']
#,'Product_Catogory','Product_Health_Info', 'Product_Display_Area', 'Branch_Area', 'Branch_Location_Type','Product_Size','Branch_Year','Branch_Type', 'source'


def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    # Perform cross-validation:
    cv_score = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))

    # Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - {:.4g} | Std - {:.4g} | Min - {:.4g} | Max - {:.4g}".format(np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

    # Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])

    # Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    #submission.to_csv(filename, index=False)

    # plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)

    plt.plot(IDcol, target)
    plt.xlabel('Value of target')
    plt.ylabel('Cross-Validated Accuracy')
    ax = plt.subplot()
    plt.show()





# Linear Regression
predictors = [x for x in train.columns if x not in [target]+IDcol]
print (predictors)
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')

# Ridge regression
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')
