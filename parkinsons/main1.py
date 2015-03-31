# -*- coding: utf-8 -*-
"""
Created on Sat Mar 07 23:57:45 2015

@author: Keshav
"""
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model


data1 = pd.read_csv("data.csv")
'''data1 = data1.drop('subject#',axis = 1)
data1 = data1.drop('age',axis = 1)
data1 = data1.drop('sex',axis = 1)
data1 = data1.drop('test_time',axis = 1)
'''
msk = np.random.rand(len(data1)) < 0.5
train =  data1[msk]
test = data1[~msk]



y = train[['motor_UPDRS','total_UPDRS']]

X = train.drop('motor_UPDRS',axis = 1)

X = X.drop('total_UPDRS',axis = 1)


labels = test[['motor_UPDRS','total_UPDRS']]
test = test.drop('motor_UPDRS',axis = 1)
test = test.drop('total_UPDRS',axis = 1)


print "Linear Regression:"
clf = LinearRegression()
clf.fit(X,y)
y_pred = clf.predict(test)
print "Rsquared value:"
print clf.score(test,labels)
print "Ceffecients:"
print clf.coef_
print "Intercept:"
print clf.intercept_
print "Mean Squared Error:"
print mean_squared_error(labels, y_pred) 
print "\n"


print "Decision Tree:"
from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf.fit(X,y)
y_pred = clf.predict(test)
print "Rsquared value:"
print clf.score(test,labels)
print "Mean Squared Error:"
print mean_squared_error(labels, y_pred) 



print "Exponential"
clf = LinearRegression()
y =np.log(y)
clf.fit(X,y)
y_pred = clf.predict(test)
print "Rsquared value:"
print clf.score(test,labels)
print "Ceffecients:"
print clf.coef_
print "Intercept:"
print clf.intercept_
print "Mean Squared Error:"
print mean_squared_error(labels, y_pred) 
print "\n"

