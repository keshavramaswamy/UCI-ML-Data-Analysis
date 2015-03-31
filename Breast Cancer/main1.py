# -*- coding: utf-8 -*-
"""
Created on Sat Mar 07 00:17:42 2015

@author: Keshav
"""

import pandas as pd
col = ['ID','a1','a2','a3','a4','a5','a6','a7','a8','a9','class']

data =pd.read_csv("data.csv",names = col,header =None)

data = data.drop('ID',axis =1)
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import tree
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
data = data[data.a6 != '?']
data.a6 = data.a6.astype(int)
msk = np.random.rand(len(data)) < 0.66
train =  data[msk]
test = data[~msk]
y = train['class']
X = train.drop('class',axis = 1)
labels = test['class']
test = test.drop('class',axis = 1)


#SVM
print "SVM"
clf = svm.SVC()
clf.fit(X,y)
print clf.score(test,labels)
y_pred = clf.predict(test)
print classification_report(y_pred,labels)

#DTree
print "DTree"
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
print clf.score(test,labels)
y_pred = clf.predict(test)
print classification_report(y_pred,labels)

#LDA
print "LDA"
clf = LDA()
clf = clf.fit(X,y)
print clf.score(test,labels)
y_pred = clf.predict(test)
print classification_report(y_pred,labels)


#NBC
print "NBC"
clf = GaussianNB()
clf = clf.fit(X,y)
print clf.score(test,labels)
y_pred = clf.predict(test)
print classification_report(y_pred,labels)


#Logistic
print "Logistic"
clf = LogisticRegression()
clf = clf.fit(X,y)
print clf.score(test,labels)
y_pred = clf.predict(test)
print classification_report(y_pred,labels)
#print y_pred


#KNN
print "KNN"
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y) 
print neigh.score(test,labels)
y_pred= neigh.predict(test)
print classification_report(y_pred,labels)