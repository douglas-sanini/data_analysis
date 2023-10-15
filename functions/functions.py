#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This code takes in all x and y values and retrieves desired interpolated values.
import scipy as sp
import numpy as np
from numpy import array
from random import uniform
from scipy.integrate import quad 
from scipy.special import erf

def gaussian_func(x: float, mu: float, sigma: float):
    #just a gaussian function
    
    return (np.exp(((-(x - mu)**2)/ (2 * (sigma**2)))))/ (sigma * (2*np.pi)**(1/2))

def gaussian_dist(x: array, mu: float, sigma: float):
    #gaussian distribution
    y = list()
    for i in range(0,len(x)):
        y.append(gaussian_func(x[i], mu, sigma))

    return y


def x_values(x: array):
    #values of x for gaussian distribution + values of x for interpolated values
    _x = list()
    for i in range(len(x)-1):
        _x = np.append(_x, x[i])
        _x = np.append(_x, np.random.uniform(x[i],x[i+1]))
    _x = np.append(_x, x[-1])
    
    return _x

def interpolated_and_other(x: array, y: list):
  #interpolated and original values
    _y = list()
    for i in range(len(y)-1):
        
        inter = y[i] + (uniform(x[i],x[i+1]) - x[i])*(y[i+1] - y[i])/(x[i+1] - x[i])
        
        _y.append(y[i])
        _y.append(inter)
    _y.append(y[-1])
    
    return _y

def interpolated(x: array, y: array):
  #interpolated values ONLY
    _y = list()
    for i in range(len(y)-1):
        
        inter = y[i] + (uniform(x[i],x[i+1]) - x[i])*(y[i+1] - y[i])/(x[i+1] - x[i])
        
        _y.append(inter)
    
    return _y

    
def cumlative(x: array, func: float, arg: tuple):
    integral = ([])
    for i in x:
        inte = quad(func,x[0],i,(arg))
        integral = np.append(integral, inte[0])
    return integral


def sampler(x_min: float, x_max: float, func: float, arg: tuple, n: int):
    x = np.linspace(x_min,x_max,n)
    _x = list()
    c = cumlative(x,func,arg)
    rand = np.random.uniform(1,0,n)
    for i in range(len(c)-1):

        for j in range(len(rand)):

            inter = x[i] + (rand[j] - c[i])*(x[i+1] - x[i])/(c[i+1] - c[i])

            if rand[j] >= c[i] and rand[j] <= c[i+1]:
                _x.append(inter)
            
            if rand[j] == c[len(c)-1]:
                _x.append(c[len(c)-1])
                
    return _x




'''
def cumlative1(x: array, func: float, arg: tuple):
    integral = array([])
    for i in range(len(x)):
        inte = quad(func,min(x),x[i],(arg))
        integral = np.append(integral, inte[0])
    return integral
'''



'''
def sampler1(x_min: float, x_max: float, func: float, arg: tuple, n: int):
    x = np.linspace(x_min,x_max,n+1)
    _x = array([])
    c = cumlative1(x,func,arg)
    rand = np.random.uniform(1,0,n)
    for i in range(len(c)-1):

        inter = x[i] + (rand[i] - c[i])*(x[i+1] - x[i])/(c[i+1] - c[i])
        
        if rand[i] >= c[i] and rand[i] <= c[i+1]:
            _x = np.append(_x, inter)
        
        else:
            for j in range(len(c)-1):

                inter2 = x[j] + (rand[i] - c[j])*(x[j+1] - x[j])/(c[j+1] - c[j])
                
                if rand[i] >= c[j] and rand[i] <= c[j+1]:
                    _x = np.append(_x, inter2)
                    break
    return _x

'''








def some_func (x: float, x_0: float, sigma: float):
    numer1 = -(x - x_0)**2
    denom1 = 2*sigma**2
    numer2 = -(x + x_0)**2
    denom2 = sigma*(2*np.pi)**(1/2)
    
    e3 = erf(x_0/(sigma*(2)**(1/2)))
    
    e1 = np.exp(numer1/denom1)/denom2 
    e2 = np.exp(numer2/denom1)/denom2
    
    return (e1-e2)/e3


