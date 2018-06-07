#importing standard libraries 
import numpy as np 
import pandas as pd 
from pandas import Series, DataFrame 

#import lightgbm and xgboost 
import lightgbm as lgb 
import xgboost as xgb

#data preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

#loading our training dataset 'adult.csv' with name 'data' using pandas
data=pd.read_csv('adult.data',header=None)


data.columns=['age','workclass','fnlwgt','education','education-num','marital_Status', 'occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week', 'native_country','Income']

#glimpse of the data
print '##############################################'
print data.head()


#Encode labels with value between 0 and n_classes-1. This creates two class labels for the Income in the dataframe.
l=LabelEncoder()
l.fit(data.Income)

#label encoding our target variable. This encodes the target variable Income.
data.Income=Series(l.transform(data.Income))
print '##############################################'
print data.Income.value_counts()

#One Hot Encoding of the Categorical features
one_hot_workclass=pd.get_dummies(data.workclass)
one_hot_education=pd.get_dummies(data.education)
one_hot_marital_Status=pd.get_dummies(data.marital_Status)
one_hot_occupation=pd.get_dummies(data.occupation)
one_hot_relationship=pd.get_dummies(data.relationship)
one_hot_race=pd.get_dummies(data.race)
one_hot_sex=pd.get_dummies(data.sex)
one_hot_native_country=pd.get_dummies(data.native_country)

########################################################################################

#removing the original categorical variables to use the encoded categorical variables
data.drop(['workclass','education','marital_Status','occupation','relationship','race','sex','native_country'],axis=1,inplace=True)

print '#############################################################'
print data


#Merging one hot encoded features with our dataset 'data'
data=pd.concat([data,one_hot_workclass,one_hot_education,one_hot_marital_Status,one_hot_occupation,one_hot_relationship, one_hot_race,one_hot_sex,one_hot_native_country],axis=1)

#looking at the columns in the modified dataframe. There are some duplicate column names mainly '?' because some categorical variables have values '?' incase of missing values. We need remove columns
print '#####################################################################'
print list(data.columns)

print '#######################################################################'
#getting indexes of the columns that have unique names.
_,i= np.unique(data.columns, return_index=True)

#selecting the unique columns using the indexes
data=data.iloc[:, i]

print '########################################################################'
print list(data.columns)

#Separating the data into features dataset x and our target dataset y
x=data.drop('Income',axis=1)
y=data.Income

y.fillna(y.mode()[0],inplace=True)

#splitting the data into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)

#The data is stored in a DMatrix object
#label is used to define our outcome variable
dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(x_test)


#setting parameters for xgboost - trying different tree depths
#parameters={'max_depth':7, 'eta':1, 'silent':1,'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05}
#parameters={'max_depth':8, 'eta':1, 'silent':1,'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05}
#parameters={'max_depth':9, 'eta':1, 'silent':1,'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05}
#parameters={'max_depth':10, 'eta':1, 'silent':1,'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05}
#parameters={'max_depth':15, 'eta':1, 'silent':1,'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05}
#parameters={'max_depth':20, 'eta':1, 'silent':1,'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05}
parameters={'max_depth':25, 'eta':1, 'silent':1,'objective':'binary:logistic','eval_metric':'auc','learning_rate':.05}
#training our model
num_round=50
from datetime import datetime
start = datetime.now()
xg=xgb.train(parameters,dtrain,num_round)


#now predicting our model on test set
ypred=xg.predict(dtest)

print '########################################################################'
print ypred

#Converting probabilities into 1 or 0
for i in range(0,len(ypred)):
    if ypred[i]>=.5:       # setting threshold to .5
       ypred[i]=1
    else:
       ypred[i]=0

print '#########################################################################'
print ypred

#calculating accuracy of our model
accuracy_xgb = accuracy_score(y_test, ypred)
mcc_xgb = matthews_corrcoef(y_test, ypred)
roc=roc_auc_score(y_test, ypred)

print '##########################################################################'
print 'accuracy = {0}'.format(accuracy_xgb)
print 'mcc = {0}'.format(mcc_xgb)
print 'roc = {0}'. format(roc)
