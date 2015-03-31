# -*- coding: utf-8 -*-
"""
Created on Sat Mar 07 20:32:48 2015

@author: Keshav
"""

import pandas as pd
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import tree
from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


data1 = pd.read_csv("pop_failures.dat",delimiter=r"\s+")
data1 = data1.drop('Study',axis = 1)
data1 = data1.drop('Run',axis = 1)
#print data1.describe()

msk = np.random.rand(len(data1)) < 0.66
train =  data1[msk]
test = data1[~msk]
print len(train)
print len(test)
y = train['outcome']
X = train.drop('outcome',axis = 1)
labels = test['outcome']
test = test.drop('outcome',axis = 1)

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
