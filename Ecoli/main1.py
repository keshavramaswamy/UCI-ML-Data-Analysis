# -*- coding: utf-8 -*-
"""
Created on Sat Mar 07 23:09:55 2015

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

col = ['name','a1','a2','a3','a4','a5','a6','a7','class']
data1 = pd.read_csv("data.txt",names= col,delimiter=r"\s+",header=None)
data1 = data1.drop('name',axis = 1)

msk = np.random.rand(len(data1)) < 0.66
print type(msk)
train =  data1[msk]
test = data1[~msk]
#print len(train)
#print len(test)
y = train['class']
X = train.drop('class',axis = 1)
labels = test['class']
test = test.drop('class',axis = 1)
'''
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

'''