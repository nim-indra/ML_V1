# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 17:40:37 2022

@author: vishwanath
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Step3.2 : Read data from source
#Best Practice to set the working directory before we
import os

print(os.getcwd())

os.chdir("D:\\ML\\Corporate\\LR\\")

print(os.getcwd())

#process the data
data = pd.read_csv("LR_Dataset.csv")

data.head()
data.shape

data.info()

data.isnull().sum()
data.isnull().sum().sum()

data.describe()

sanitychecks = data.describe().T
print(sanitychecks)
data.describe().T.to_csv("sanitychecks.csv")


#correlation
correlations = data.corr()
sns.heatmap(correlations, annot=True)
# data.corr().to_csv("corr.csv")
correlations.to_csv("corr.csv")

## Below will be useful when feature datatype is Categorical
## data["airline_test"] = data["airline"].cat.codes


"""
data['airline'].replace({'Air_India':1,'AirAsia':2,'GO_FIRST':3,'Indigo':4,'SpiceJet':5,'Vistara':6},inplace=True)

data['stops'].replace({'one':1,'two_or_more':2,'zero':3},inplace=True)
data['arrival_time'].replace({'Afternoon':1,'Early_Morning':2,'Evening':3,'Late_Night':4,'Morning':5,'Night':6},inplace=True)

data['source_city'].replace({'Bangalore':1,'Chennai':2,'Delhi':3,'Hyderabad':4,'Kolkata':5,'Mumbai':6},inplace=True)
data['departure_time'].replace({'Afternoon':1,'Early_Morning':2,'Evening':3,'Late_Night':4,'Morning':5,'Night':6},inplace=True)

data['destination_city'].replace({'Bangalore':1,'Chennai':2,'Delhi':3,'Hyderabad':4,'Kolkata':5,'Mumbai':6},inplace=True)

data['class'].replace({'Business':1,'Economy':2},inplace=True)

"""


data=data.drop(data.columns[[0,2]],axis=1)


########################################################
################### function to detect outliers
def detect_outliers_iqr(data1):
    outliers = []
    data1 = sorted(data1)
    q1 = np.percentile(data1, 25)
    q3 = np.percentile(data1, 75)
    # print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    # print(lwr_bound, upr_bound)
    for i in data1: 
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
    return outliers# Driver code

duration_outliers = detect_outliers_iqr(data['duration'])
print("Duration Outliers from IQR method: ", duration_outliers)


fig = plt.figure(figsize =(10,1))

#dfp10 = pd.DataFrame(data = np.random.random(size=(20,2)), columns = ['duration','days_left'])

dfp10 = pd.DataFrame(data = np.random.random(size=(10,1)), columns = ['duration'])

dfp10.boxplot()



data.dtypes

##final_iv, IV = data(data , data.price)


feature=['airline','source_city','departure_time','stops','arrival_time','destination_city','class','duration','days_left']
target = 'price'
df_woe_iv = (pd.crosstab(data[feature],data[target],
                      normalize='columns')
             .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]))
             .assign(iv=lambda dfx: np.sum(dfx['woe']*
                                           (dfx[1]-dfx[0]))))

df_woe_iv




data = pd.get_dummies(data=data, columns=['airline','source_city','departure_time','stops','arrival_time','destination_city','class'])
data.columns


data.drop(['airline','source_city','departure_time','stops','arrival_time','destination_city','class'],axis=1,inplace=True)

data.info()

data.to_csv("check.csv")


from sklearn.model_selection import train_test_split
# Define our predictor and target variables
#X = data.drop(['price','Unnamed:0'],axis=1)

# X=data['airline','flight','source_city','departure_time','stops','arrival_time','destination_city','class']
# X = data.drop(labels=['price','Unnamed'], axis=1)

##data.drop(['Unnamed: 0'],inplace=True,axis=1)
##data.drop(['Unnamed: 0'],inplace=True,axis=1)
##data.shape
X = data.drop(labels=['price'], axis=1)

list(X)
# np.shape(X)

y = data['price']
list(y)
np.shape(y)

# Create four groups using train_test_split. By default, 75% of data is assigned to train, the other 25% to test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=2)

print(X_train)
print(X_test)

print(y_train)
print(y_test)


from sklearn.linear_model import LinearRegression
# Initialize a linear regression model object
lm = LinearRegression()
# Fit the linear regression model object to our data
lm.fit(X_train,y_train)






#Predictions from our Model
#Let's grab predictions off our Train set and see how well it did!
y_pred=lm.predict(X_train)


# import numpy as np 
import pylab 
import scipy.stats as stats
resid=y_train-y_pred
stats.probplot(resid, dist="norm", plot=pylab)
pylab.show()




resid=y_pred-y_train
#resid=pd.Series(resid)
plt.scatter(y_pred,resid)
plt.hlines(0,0,12000)

# resid=y_pred-y_train
# #resid=pd.Series(resid)
# plt.scatter(y_pred,resid)
# plt.hlines(0,0,12000)

from sklearn.metrics import r2_score
#VIF = 1/1-r2
VIF = 1/ (1-r2_score(y_train,y_pred))
print(VIF)

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif["features"] = X.columns
print(vif["VIF Factor"])



# model summary on Train
import statsmodels.api as sm
X1=sm.add_constant(X_train)
est=sm.OLS(y_train,X1)
est2=est.fit()
print(est2.summary())



y_pred_test=lm.predict(X_test)

# model summary
import statsmodels.api as sm
X2=sm.add_constant(X_test)
est_t=sm.OLS(y_test,X2)
est2_t=est_t.fit()
print(est2_t.summary())










