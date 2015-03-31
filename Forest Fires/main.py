# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 16:13:29 2015

@author: Rithi
"""


import pandas as pd
bc_data=pd.read_csv("forestfires.csv")


month=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
day=['sun','mon','tue','wed','thu','fri','sat']
col1=bc_data['month']
c=0
for iter in col1:
    for iter1 in month:
        if iter==iter1:
            col1[c]=month.index(iter1)
    c=c+1           
bc_data['month']=col1

col2=bc_data['day']
c=0
for iter in col2:
    for iter1 in day:
        if iter==iter1:
            col2[c]=day.index(iter1)
    c=c+1           
bc_data['day']=col2



bc_data['month']=bc_data['month'].astype(int)
bc_data['day']=bc_data['day'].astype(int)



import numpy as np
msk = np.random.rand(len(bc_data)) < 0.66
train = bc_data[msk]
test = bc_data[~msk]


y=train['area']
del train['area']


#linear regression
from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit (train,y)
print "Linear Regression"
y1=test['area']
del test['area']
y2=clf.predict(test)
from sklearn.metrics import mean_squared_error
print "Rsquared value",clf.score(test,y1)
print "coefficients",clf.coef_
print "mean squared error",mean_squared_error(y1,y2)

#decision tree
from sklearn import tree
print "Decision Tree"
clf = tree.DecisionTreeRegressor()
clf.fit(train,y)
y2 = clf.predict(test)
print "Rsquared value",clf.score(test,y1)
print "mean squared error",mean_squared_error(y1,y2)


#exponential regression
clf = linear_model.LinearRegression()
clf.fit (train,np.log(y+1))
y2 = clf.predict(test)
print "Exponential Regression"
print "Rsquared value",clf.score(test,y1)
print "coefficients",clf.coef_
print "mean squared error",mean_squared_error(y1,y2) 

