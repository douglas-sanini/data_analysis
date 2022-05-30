#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This code takes in all x and y values and retrieves desired interpolated values.
import scipy as sp
import numpy as np
from numpy import array
from random import uniform


def gaussian_func(x: float, mu: float, sigma: float):
    #just a gaussian function
    
    return (np.exp(((-(x - mu)**2)/ 2 * sigma**2))/(sigma * (2*np.pi)**(1/2)))

def gaussian_dist(x: array, mu: float, sigma: float):
    #gaussian distribution
    y = array([])
    for i in range(0,len(x)):
        y = np.append(y, gaussian_func(x[i], mu, sigma))

    return y


def x_values(x: array):
    #values of x for gaussian distribution + values of x for interpolated values
    _x = array([])
    for i in range(len(x)-1):
        _x = np.append(_x, x[i])
        _x = np.append(_x, np.random.uniform(x[i],x[i+1]))
    _x = np.append(_x, x[-1])
    
    return _x

def interpolated_and_other(x: array, y: array):
  #interpolated and original values
    _y = array([])
    for i in range(len(y)-1):
        
        inter = y[i] + (uniform(x[i],x[i+1]) - x[i])*(y[i+1] - y[i])/(x[i+1] - x[i])
        
        _y = np.append(_y, y[i])
        _y = np.append(_y, inter)
    _y = np.append(_y, y[-1])
    
    return _y

def interpolated(x: array, y: array):
  #interpolated values ONLY
    _y = array([])
    for i in range(len(y)-1):
        
        inter = y[i] + (uniform(x[i],x[i+1]) - x[i])*(y[i+1] - y[i])/(x[i+1] - x[i])
        
        _y = np.append(_y, inter)
    
    return _y

    

