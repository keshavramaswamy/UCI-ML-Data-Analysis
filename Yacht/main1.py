# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 13:18:27 2015

@author: Keshav
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error






col=[]
for iter in range(0,6):
    col.append('p'+str(iter))
col.append('output')
#print col
data =pd.read_csv("data.txt",header =None,delimiter=r"\s+",names=col)

'''for iter in col:
    print iter
    print data[iter].value_counts()'''
    

#print data

msk = np.random.rand(len(data)) < 0.7
train =  data[msk]
test = data[~msk]
y = train['output']
X = train.drop('output',axis = 1)
labels = test['output']
test = test.drop('output',axis = 1)

#LinRegression
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


print "\n"
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



