# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 11:54:13 2015

@author: Keshav
"""

import numpy as np
from scipy.optimize import curve_fit
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
    
xdata = np.linspace(0, 4, 50)

y = func(xdata, 2.5, 1.3, 0.5)

print xdata

print y

ydata = y + 0.2 * np.random.normal(size=len(xdata))

popt, pcov = curve_fit(func, xdata, ydata)

print popt