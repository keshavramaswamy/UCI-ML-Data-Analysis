# -*- coding: utf-8 -*-
"""
Created on Sat Mar 07 18:55:14 2015

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

col=[]

for iter in range(0,64):
    print iter
    col.append(str(iter))
    
col.append('class')


print col


traindata =pd.read_csv("optdigitstra.csv",names=col,header=None)
testdata = pd.read_csv("optdigitstes.csv",names=col,header=None)

#print traindata['class']  
y = traindata['class']
X = traindata.drop('class',axis = 1)
labels = testdata['class']
test = testdata.drop('class',axis = 1)

#print traindata['class'].value_counts()


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