# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 16:08:44 2015

@author: Keshav
"""

import pdb, csv
import numpy as np
from sklearn import preprocessing

dataset = np.genfromtxt('forestfires.csv', delimiter=',', skip_header=1)
np.random.shuffle(dataset)

standard_scaler = preprocessing.StandardScaler()

xs_scale = standard_scaler.fit_transform(dataset[:][:,4:12])
ys = dataset[:][:,12]

ys[ys == 0.0] = -1
ys[ys != -1] = 1

dataset_processed = np.c_[ys, xs_scale]

np.savetxt("forestfires_processed.csv", dataset_processed, delimiter=",")