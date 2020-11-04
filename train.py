import pandas as pd
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.impute import SimpleImputer;
from sklearn.compose import ColumnTransformer;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression ;
from sklearn.linear_model import Ridge, Lasso;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import r2_score;
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.svm import SVR;
from sklearn.svm import SVC;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.naive_bayes import GaussianNB;
import xgboost as xgb;
from xgboost import XGBClassifier;
from xgboost import XGBRegressor;
from lightgbm import LGBMRegressor
import pickle;

train = pd.read_csv('../input/big-mart-sales-dataset/Train_UWu5bXk.csv');
test = pd.read_csv('../input/big-mart-sales-dataset/Test_u94Q5KV.csv')

train.head(30)

test.head()

train.isnull().sum()/train.shape[0]*100

test.isnull().sum()/test.shape[0]*100

train['Outlet_Size'].mode()

train['Outlet_Size'].unique()

train.update(train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0]));
test.update(test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0]));

train.head()

train.isnull().sum()/train.shape[0]*100

for var in train['Item_Type'].unique():
    train.update(train[train.loc[:,'Item_Type'] == var]['Item_Weight'].replace(np.nan,train[train.loc[:,'Item_Type'] == var]['Item_Weight'].mean()));

for var in test['Item_Type'].unique():
    test.update(test[test.loc[:,'Item_Type'] == var]['Item_Weight'].replace(np.nan,test[test.loc[:,'Item_Type'] == var]['Item_Weight'].mean()));

test.isnull().sum()/test.shape[0]*100

train = train.drop(columns = ['Item_Identifier', 'Outlet_Identifier'], axis =1);
test = test.drop(columns = ['Item_Identifier', 'Outlet_Identifier'], axis =1);

train['Item_Fat_Content'].unique()

f = {'Low Fat': 1, 'Regular': 2, 'low fat': 3, 'LF': 4, 'reg': 5};
train['Item_Fat_Content'] = train['Item_Fat_Content'].map(f);
test['Item_Fat_Content'] = test['Item_Fat_Content'].map(f)

train['Item_Type'].unique()

t = {'Dairy': 1, 'Soft Drinks': 2, 'Meat': 3, 'Fruits and Vegetables' : 4,
       'Household': 5, 'Baking Goods': 6, 'Snack Foods': 7, 'Frozen Foods': 8,
       'Breakfast': 9, 'Health and Hygiene' :10, 'Hard Drinks': 11, 'Canned': 12,
       'Breads': 13, 'Starchy Foods': 15, 'Others': 14, 'Seafood': 16};
train['Item_Type'] = train['Item_Type'].map(t);
test['Item_Type'] = test['Item_Type'].map(t)

s = {'Medium': 2, 'High': 3, 'Small': 1};
train['Outlet_Size'] = train['Outlet_Size'].map(s);
test['Outlet_Size'] = test['Outlet_Size'].map(s)

train.head()

train['Outlet_Location_Type'].unique()

l = {'Tier 1': 1, 'Tier 3': 3, 'Tier 2': 2};
train['Outlet_Location_Type'] = train['Outlet_Location_Type'].map(l);
test['Outlet_Location_Type'] = test['Outlet_Location_Type'].map(l)

train['Outlet_Type'].unique()

y = {'Supermarket Type1': 2, 'Supermarket Type2': 3, 'Grocery Store': 1,
       'Supermarket Type3': 4};
train['Outlet_Type'] = train['Outlet_Type'].map(y);
test['Outlet_Type'] = test['Outlet_Type'].map(y)

train.head(8)

plt.figure(figsize = (16,9))
fineTech_appData3 = train.drop(['Item_Outlet_Sales'], axis = 1) # drop 'enrolled' feature
sns.barplot(fineTech_appData3.columns,fineTech_appData3.corrwith(train['Item_Outlet_Sales']))

train = train.drop(columns = ['Item_Weight', 'Item_Fat_Content', 'Item_Type', 'Outlet_Establishment_Year' ], axis =1);
test = test.drop(columns = ['Item_Weight', 'Item_Fat_Content', 'Item_Type', 'Outlet_Establishment_Year' ], axis =1);

train.head()

x = train.drop(columns  = ['Item_Outlet_Sales'], axis =1);
y = train['Item_Outlet_Sales']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 55)

rfc = RandomForestRegressor();
rfc.fit(xtrain, ytrain)

rfc.score(xtest, ytest)

xgb = XGBRegressor();
xgb.fit(xtrain, ytrain)

xgb.score(xtest, ytest)

knn = KNeighborsRegressor();
knn.fit(xtrain, ytrain)

knn.score(xtest, ytest);

dc= DecisionTreeRegressor();
dc.fit(xtrain, ytrain);

dc.score(xtest, ytest)
