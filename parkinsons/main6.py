# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 14:50:50 2015

@author: Keshav
"""

import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model


data1 = pd.read_csv("data.csv")
msk = np.random.rand(len(data1)) < 0.66
train =  data1[msk]
test = data1[~msk]
y = train[['motor_UPDRS','total_UPDRS']]
X = train.drop('motor_UPDRS',axis = 1)
X = X.drop('total_UPDRS',axis = 1)
labels = test[['motor_UPDRS','total_UPDRS']]
test = test.drop('motor_UPDRS',axis = 1)
test = test.drop('total_UPDRS',axis = 1)
clf = linear_model.BayesianRidge()
clf.fit(X,y)