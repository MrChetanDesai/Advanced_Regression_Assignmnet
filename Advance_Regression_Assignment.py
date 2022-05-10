#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[149]:


#import libraries 
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

#import train_test_split to split the data
from sklearn.model_selection import train_test_split

#Scaling using MinMax
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

#import libraries for model evalution
from sklearn.metrics import r2_score, mean_squared_error

#Ridge and lasso regression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

import os
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# ## Data Understanding 
# 

# In[79]:


#Reading the data set

data = pd.read_csv("train.csv", encoding = 'utf-8')
data.head()


# In[80]:


#Understand the data

data.info()


# In[81]:


# Check the dimensions

data.shape


# In[82]:


# The description of the dataset

data.describe()


# In[83]:


#checking duplicates

sum(data.duplicated(subset = 'Id')) == 0


# In[84]:


#Sum of null value

data.isnull().sum()


# In[85]:


null = pd.DataFrame(round(data.isnull().sum()/len(data.index)*100,2).sort_values(ascending=False),columns=["Null %"])
null.index.name = 'Features'
null.head()


# In[86]:


# dataframe with features having null values

null_df = null[null["Null %"] > 0]
null_df


# In[87]:


# we will drop the 'PoolQC','MiscFeature','Alley','Fence','FireplaceQu'column becuase there are so many missing values and id column is not required

data = data.drop(['PoolQC','MiscFeature','Id','Alley','Fence','FireplaceQu'],axis=1)


# In[88]:


null = pd.DataFrame(round(data.isnull().sum()/len(df.index)*100,2).sort_values(ascending=False),columns=["Null %"])
null.index.name = 'Features'
null_df = null[null["Null %"] > 0]
null_df


# In[89]:


data.columns


# In[90]:


#Categorical columns

data.select_dtypes(include='object').columns


# In[91]:


# Numeric columns

data.select_dtypes(exclude='object').columns


# In[92]:


# for the LotFrontage column and GarageYrBlt we will impute the missing values with the median since the feature contains outliers

data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].median())
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['GarageYrBlt'].median())

# for the "below columns" we will impute the null values with 'mode'
for col in ('GarageCond', 'GarageType', 'GarageFinish','GarageQual'):
    data[col] = data[col].fillna(data[col].mode()[0])
    
# for the "Bsmt" columns we will impute the null values with 'mode'
for col in ('BsmtExposure', 'BsmtFinType2', 'BsmtFinType1','BsmtCond','BsmtQual'):    
    data[col] = data[col].fillna(data[col].mode()[0])
    
# for the columns we will impute the null values with 'mode'
for col in ('MasVnrArea', 'MasVnrType', 'Electrical'):
    data[col] = data[col].fillna(data[col].mode()[0])


# In[93]:


null = pd.DataFrame(round(data.isnull().sum()/len(data.index)*100,2).sort_values(ascending=False),columns=["Null %"])
null.index.name = 'Features'
null_df = null[null["Null %"] > 0]
null_df


# In[94]:


# checking for the presence of any more null values

data.isnull().values.any()


# In[95]:


# check Null value

data.isnull().sum()


# In[96]:


# Check the shape

data.shape


# In[97]:


data.describe()


# In[98]:


print(data['PoolArea'].value_counts())
print(data['MiscVal'].value_counts())
print(data['3SsnPorch'].value_counts())


# In[99]:


# we will drop these columns as it dominated by one value and it won't add any extra information to our model

data = data.drop(['PoolArea','MiscVal','3SsnPorch'],axis=1)


# In[100]:


data.shape


# In[101]:


data.describe()


# ## Visualization of Data

# In[103]:


# Sale columns
plt.figure()
sns.distplot(df['SalePrice'],color='r')
plt.show()


# In[104]:


# Numeric columns

df.select_dtypes(exclude='object').columns


# In[105]:


# Sale columns

plt.figure()
sns.distplot(df['GrLivArea'],color='g')
plt.show()


# In[106]:


# Remove the outlier
cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
         'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
        'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch','ScreenPorch', 'MoSold', 'YrSold', 'SalePrice'] # one or more

Q1 = df[cols].quantile(0.05)
Q3 = df[cols].quantile(0.95)
IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[107]:


# Sale columns

plt.figure()
sns.distplot(df['SalePrice'],color='b')
plt.show()


# In[108]:


data.info()

# after we remove some outliers


# In[109]:


fig, ax = plt.subplots(ncols=6, sharey=True)

MSSubClass = ax[0].scatter(data['MSSubClass'], data['SalePrice'], color='red')
MSSubClass.set_label("MSSubClass")
ax[0].set_title('MSSubClass vs Sales')

LotFrontage= ax[1].scatter(data['LotFrontage'], data['SalePrice'], color='green')
LotFrontage.set_label("LotFrontage")
ax[1].set_title('LotFrontage vs Sales')

LotArea = ax[2].scatter(data['LotArea'], data['SalePrice'], color='blue')
LotArea.set_label("LotArea")
ax[2].set_title('LotArea vs Sales')

OverallQual = ax[3].scatter(data['OverallQual'], data['SalePrice'],color='red')
OverallQual.set_label("LotArea")
ax[3].set_title('OverallQual vs Sales')

OverallCond = ax[4].scatter(data['OverallCond'], data['SalePrice'],color='green')
OverallCond.set_label("OverallCond")
ax[4].set_title('OverallCond vs Sales')

YearBuilt = ax[5].scatter(data['YearBuilt'], data['SalePrice'],color='blue')
YearBuilt.set_label("YearBuilt")
ax[5].set_title('YearBuilt vs Sales')


fig.set_size_inches(20.5, 10.5, forward=True)

plt.show()


# In[65]:


fig, ax = plt.subplots(ncols=6, sharey=True)

YearRemodAdd = ax[0].scatter(data['YearRemodAdd'], data['SalePrice'], color='red')
YearRemodAdd.set_label("YearRemodAdd")
ax[0].set_title('YearRemodAddvs Sales')

MasVnrArea= ax[1].scatter(data['MasVnrArea'], data['SalePrice'], color='green')
MasVnrArea.set_label("MasVnrArea")
ax[1].set_title('MasVnrArea vs Sales')

BsmtFinSF1 = ax[2].scatter(data['BsmtFinSF1'], data['SalePrice'], color='blue')
BsmtFinSF1.set_label("BsmtFinSF1")
ax[2].set_title('BsmtFinSF1 vs Sales')

BsmtFinSF2 = ax[3].scatter(data['BsmtFinSF2'], data['SalePrice'], color='red')
BsmtFinSF2.set_label("BsmtFinSF2")
ax[3].set_title('BsmtFinSF2 vs Sales')

BsmtUnfSF = ax[4].scatter(data['BsmtUnfSF'], data['SalePrice'], color='green')
BsmtUnfSF.set_label("BsmtUnfSF")
ax[4].set_title('BsmtUnfSF vs Sales')

TotalBsmtSF = ax[5].scatter(data['TotalBsmtSF'], data['SalePrice'], color='blue')
TotalBsmtSF.set_label("TotalBsmtSF")
ax[5].set_title('TotalBsmtSF vs Sales')


fig.set_size_inches(20.5, 10.5, forward=True)


# In[110]:


fig, ax = plt.subplots(ncols=6, sharey=True)

stFlrSF = ax[0].scatter(data['1stFlrSF'], data['SalePrice'], color='red')
stFlrSF.set_label("1stFlrSF")
ax[0].set_title('1stFlrSF vs Sales')

ndFlrSF= ax[1].scatter(data['2ndFlrSF'], data['SalePrice'], color='green')
ndFlrSF.set_label("2ndFlrSF")
ax[1].set_title('2ndFlrSF vs Sales')

LowQualFinSF = ax[2].scatter(data['LowQualFinSF'], data['SalePrice'], color='blue')
LowQualFinSF.set_label("LowQualFinSF")
ax[2].set_title('LowQualFinSF vs Sales')

GrLivArea = ax[3].scatter(data['GrLivArea'], data['SalePrice'], color='red')
GrLivArea.set_label("GrLivArea")
ax[3].set_title('GrLivArea vs Sales')

BsmtFullBath = ax[4].scatter(data['BsmtFullBath'], data['SalePrice'], color='green')
BsmtFullBath.set_label("BsmtFullBath")
ax[4].set_title('BsmtFullBath vs Sales')

BsmtHalfBath = ax[5].scatter(data['BsmtHalfBath'], data['SalePrice'], color='blue')
BsmtHalfBath.set_label("BsmtHalfBath")
ax[5].set_title('BsmtHalfBath vs Sales')


fig.set_size_inches(20.5, 10.5, forward=True)

plt.show()


# In[50]:


fig, ax = plt.subplots(ncols=6, sharey=True)

FullBath = ax[0].scatter(data['FullBath'], data['SalePrice'], color='red')
FullBath.set_label("FullBath")
ax[0].set_title('FullBath vs Sales')

HalfBath= ax[1].scatter(data['HalfBath'], data['SalePrice'], color='green')
HalfBath.set_label("HalfBath")
ax[1].set_title('HalfBath vs Sales')

BedroomAbvGr = ax[2].scatter(data['BedroomAbvGr'], data['SalePrice'], color='blue')
BedroomAbvGr.set_label("BedroomAbvGr")
ax[2].set_title('BedroomAbvGr vs Sales')

KitchenAbvGr = ax[3].scatter(data['KitchenAbvGr'], data['SalePrice'], color='red')
KitchenAbvGr.set_label("KitchenAbvGr")
ax[3].set_title('KitchenAbvGr vs Sales')

TotRmsAbvGrd = ax[4].scatter(data['TotRmsAbvGrd'], data['SalePrice'], color='green')
TotRmsAbvGrd.set_label("TotRmsAbvGrd")
ax[4].set_title('TotRmsAbvGrd vs Sales')

Fireplaces = ax[5].scatter(data['Fireplaces'], data['SalePrice'], color='blue')
Fireplaces.set_label("Fireplaces")
ax[5].set_title('Fireplaces vs Sales')


fig.set_size_inches(20.5, 10.5, forward=True)

plt.show()


# In[111]:


fig, ax = plt.subplots(ncols=6, sharey=True)

GarageYrBlt = ax[0].scatter(data['GarageYrBlt'], data['SalePrice'], color='red')
GarageYrBlt.set_label("GarageYrBlt")
ax[0].set_title('GarageYrBlt vs Sales')

GarageCars= ax[1].scatter(data['GarageCars'], data['SalePrice'] , color='green')
GarageCars.set_label("GarageCars")
ax[1].set_title('GarageCars vs Sales')

GarageArea = ax[2].scatter(data['GarageArea'], data['SalePrice'],  color='blue')
GarageArea.set_label("GarageArea")
ax[2].set_title('GarageArea vs Sales')

WoodDeckSF = ax[3].scatter(data['WoodDeckSF'], data['SalePrice'], color='red')
WoodDeckSF.set_label("WoodDeckSF")
ax[3].set_title('WoodDeckSF vs Sales')

OpenPorchSF = ax[4].scatter(data['OpenPorchSF'], data['SalePrice'],  color='green')
OpenPorchSF.set_label("OpenPorchSF")
ax[4].set_title('OpenPorchSF vs Sales')

EnclosedPorch = ax[5].scatter(data['EnclosedPorch'], data['SalePrice'], color='blue')
EnclosedPorch.set_label("EnclosedPorch")
ax[5].set_title('EnclosedPorch vs Sales')


fig.set_size_inches(20.5, 10.5, forward=True)

plt.show()


# In[112]:


fig, ax = plt.subplots(ncols=3, sharey=True)


ScreenPorch= ax[0].scatter(data['ScreenPorch'], data['SalePrice'], color='red')
ScreenPorch.set_label("ScreenPorch")
ax[0].set_title('ScreenPorch vs Sales')

MoSold = ax[1].scatter(data['MoSold'], data['SalePrice'], color='green')
MoSold.set_label("MoSold")
ax[1].set_title('MoSoldvs Sales')

YrSold = ax[2].scatter(data['YrSold'], data['SalePrice'], color='blue')
YrSold.set_label("YrSold")
ax[2].set_title('YrSold vs Sales')


fig.set_size_inches(20.5, 10.5, forward=True)

plt.show()


# In[113]:


#Categorical columns

data.select_dtypes(include='object').columns


# In[114]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(data.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[115]:


# sales price correlation matrix
plt.figure(figsize = (16, 10))
n = 25 # number of variables which have the highest correlation with 'Sales price'

corrmat = data.corr()
cols = corrmat.nlargest(n, 'SalePrice')['SalePrice'].index

#plt.figure(dpi=100)
sns.heatmap(data[cols].corr(),annot=True)
plt.show()


# ## Dummy Variable

# In[116]:


#Categorical columns

data.select_dtypes(include='object').columns


# In[117]:


# Convert categorical value into Dummy variable

data=pd.get_dummies(data,drop_first=True)
data.head()


# ## Splitting the data into train and test sets

# In[118]:


#Split the data into train and test

y = data.pop('SalePrice')
y.head()


# In[119]:


X = data
X.shape


# In[120]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[121]:


X_train.head()


# In[122]:


print('X_train shape',X_train.shape)
print('X_test shape',X_test.shape)
print('y_train shape',y_train.shape)
print('y_test shape',y_test.shape)


# ## Scaling of numeric varaibles

# In[123]:


X_train.head()


# In[124]:


y_train.head()


# In[125]:


X_test.head()


# In[126]:


y_test.head()


# In[127]:


# columns to be scaled

X_train.select_dtypes(include=['int64','int32','float64','float32']).columns


# In[128]:


num_vars= ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
           'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
           '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
           'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
           'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'MoSold', 'YrSold']
X_train[num_vars].head()


# In[129]:


X_train.describe()


# In[130]:


X_train.head()


# In[133]:



#scaler = StandardScaler()
scaler=MinMaxScaler()


# In[134]:


X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
X_test[num_vars] = scaler.transform(X_test[num_vars])


# In[135]:


X_train.head()


# In[136]:


X_test.head()


# In[137]:


X_train.describe()


# In[138]:


X_train.shape


# ## Model Building

# ### RFE

# In[140]:


# Running RFE with the output number of the variable equal to 10

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 25)             # running RFE
rfe = rfe.fit(X_train, y_train)


# In[141]:


#Find the top features

list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[142]:


# Extract the top features

col = X_train.columns[rfe.support_]
col


# In[143]:


# Extract the non-important features

X_train.columns[~rfe.support_]


# In[144]:


#Check the shape of train and test

X_train1=X_train[col]
X_test1=X_test[col]
print(X_train1.shape)
print(X_test1.shape)
print(y_train.shape)
print(y_test.shape)


# In[145]:


lm1=lm.fit(X_train, y_train)


# In[146]:


# Print the coefficients and intercept

print(lm1.intercept_)
print(lm1.coef_)


# In[148]:


#r2score,RSS and RMSE

y_pred_train = rfe.predict(X_train)
y_pred_test = rfe.predict(X_test)

metric = []
r2_train_lr = r2_score(y_train, y_pred_train)
print(r2_train_lr)
metric.append(r2_train_lr)

r2_test_lr = r2_score(y_test, y_pred_test)
print(r2_test_lr)
metric.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train - y_pred_train))
print(rss1_lr)
metric.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test - y_pred_test))
print(rss2_lr)
metric.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train, y_pred_train)
print(mse_train_lr)
metric.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print(mse_test_lr)
metric.append(mse_test_lr**0.5)


# ## Ridge Regression

# In[150]:


params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}


ridge = Ridge()

# cross validation
folds = 5
ridge_model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
ridge_model_cv.fit(X_train1, y_train) 


# In[151]:


print(ridge_model_cv.best_params_)
print(ridge_model_cv.best_score_)


# In[153]:


#Optimim value of alpha is 1.0

alpha = 1
ridge = Ridge(alpha=alpha)
ridge.fit(X_train1, y_train)
ridge.coef_


# In[154]:


# Lets calculate some metrics such as R2 score, RSS and RMSE
y_pred_train = ridge.predict(X_train1)
y_pred_test = ridge.predict(X_test1)

metric2 = []
r2_train_lr = r2_score(y_train, y_pred_train)
print(r2_train_lr)
metric2.append(r2_train_lr)

r2_test_lr = r2_score(y_test, y_pred_test)
print(r2_test_lr)
metric2.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train - y_pred_train))
print(rss1_lr)
metric2.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test - y_pred_test))
print(rss2_lr)
metric2.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train, y_pred_train)
print(mse_train_lr)
metric2.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print(mse_test_lr)
metric2.append(mse_test_lr**0.5)


# ## Lasso Regression

# In[155]:


lasso = Lasso()

# cross validation
lasso_model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

lasso_model_cv.fit(X_train1, y_train)


# In[156]:


print(lasso_model_cv.best_params_)
print(lasso_model_cv.best_score_)


# In[157]:


alpha =50

lasso = Lasso(alpha=alpha)
        
lasso.fit(X_train1, y_train) 


# In[158]:


lasso.coef_


# In[159]:


# Lets calculate some metrics such as R2 score, RSS and RMSE

y_pred_train = lasso.predict(X_train1)
y_pred_test = lasso.predict(X_test1)

metric3 = []
r2_train_lr = r2_score(y_train, y_pred_train)
print(r2_train_lr)
metric3.append(r2_train_lr)

r2_test_lr = r2_score(y_test, y_pred_test)
print(r2_test_lr)
metric3.append(r2_test_lr)

rss1_lr = np.sum(np.square(y_train - y_pred_train))
print(rss1_lr)
metric3.append(rss1_lr)

rss2_lr = np.sum(np.square(y_test - y_pred_test))
print(rss2_lr)
metric3.append(rss2_lr)

mse_train_lr = mean_squared_error(y_train, y_pred_train)
print(mse_train_lr)
metric3.append(mse_train_lr**0.5)

mse_test_lr = mean_squared_error(y_test, y_pred_test)
print(mse_test_lr)
metric3.append(mse_test_lr**0.5)


# In[160]:


metric2


# In[161]:


# Creating a table which contain all the metrics

lr_table = {'Metric': ['R2 Score (Train)','R2 Score (Test)','RSS (Train)','RSS (Test)',
                       'MSE (Train)','MSE (Test)'], 
        'Linear Regression': metric
        }

lr_metric = pd.DataFrame(lr_table ,columns = ['Metric', 'Linear Regression'] )

rg_metric = pd.Series(metric2, name = 'Ridge Regression')
ls_metric = pd.Series(metric3, name = 'Lasso Regression')

final_metric = pd.concat([lr_metric, rg_metric, ls_metric], axis = 1)

final_metric


# ## Model Evaluation

# In[162]:


ridge_pred = ridge.predict(X_test1)


# In[163]:


# Plotting y_test and y_pred to understand the spread for ridge regression.

fig = plt.figure(dpi=100)
plt.scatter(y_test,ridge_pred)
fig.suptitle('y_test vs ridge_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('ridge_pred', fontsize=16)  
plt.show()


# In[164]:


y_res=y_test-ridge_pred
# Distribution of errors
sns.distplot(y_res,kde=True)
plt.title('Normality of error terms/residuals')
plt.xlabel("Residuals")
plt.show()


# In[165]:


lasso_pred = lasso.predict(X_test1)


# In[166]:


# Plotting y_test and y_pred to understand the spread for lasso regression.

fig = plt.figure(dpi=100)
plt.scatter(y_test,lasso_pred)
fig.suptitle('y_test vs lasso_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('lasso_pred', fontsize=16)  
plt.show()


# In[167]:


y_res=y_test-lasso_pred
# Distribution of errors
sns.distplot(y_res,kde=True)
plt.title('Normality of error terms/residuals')
plt.xlabel("Residuals")
plt.show()


# ## changes in the coefficients after regularization

# In[168]:


betas = pd.DataFrame(index=X_train1.columns)
betas.rows = X_train1.columns


# In[169]:


betas['Ridge'] = ridge.coef_
betas['Lasso'] = lasso.coef_


# In[170]:


pd.set_option('display.max_rows', None)
betas.head(68)


# ### Question 1-Which variables are significant in predicting the price of a house?

# ### Answer:
#     *As per the analyses of the model following are the variables in predicting the house of the price
# 
#     
#     *LotArea------------- Lot size in square feet
#     *OverallQual--------- Rates the overall material and finish of the house
#     *OverallCond--------- Rates the overall condition of the house
#     *YearBuilt----------- Original construction date
#     *BsmtFinSF1---------- Type 1 finished square feet
#     *TotalBsmtSF--------- Total square feet of basement area
#     *GrLivArea----------- Above grade (ground) living area square feet
#     *TotRmsAbvGrd-------- Total rooms above grade (does not include bathrooms)
#     *Street_Pave--------- Pave road access to property
#     *RoofMatl_Metal------ Roof material_Metal
# 
# 
# 

# ## Q2 -How well those variables describe the price of a house

#              Ridge Regression --------------- Lasso Regression
# R2 score(Train)---------------- 0.88 -------------------------------0.88
# 
# R2 score(Test)----------------- 0.87--------------------------------0.86

# In[173]:


final_metric


# In[ ]:




