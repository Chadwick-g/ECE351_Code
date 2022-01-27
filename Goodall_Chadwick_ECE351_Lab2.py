#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################
# #
# Chadwick Goodall #
# ECE 351 Section 53 #
# Lab 2 #
# 2-1-22 #
################################################################

import numpy as np
import matplotlib.pyplot as plt



#plt.rcParams.update({'fontsize': 14})
steps = 1e-2
t = np.arange(0, 1+steps,steps) 
 
def func1(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        y[i] = np.cos(t[i])
    return y
    
#y = func1(t)



# beginning of the step function and plot

def u(t):
    y = np.zeros(t.shape)

    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
            
    return y
    
#y = u(t)


# beginning of the ramp function and plot
  
def r(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
        
    return y

#y = r(t)
    
# beginning of composite function using rmp and step
t = np.arange(-5, 10+steps, steps)   
def customPlt(t):
    #y = np.zeros(t.shape)
    
    y = r(t) - r(t - 3) + 5 * u(t - 3) - 2 * u(t - 6) - 2 * r(t - 6)

        
    return y
    
y = customPlt(t)

def ddt(y):

    dy = np.diff(y, axis=0)
    dt = np.diff(t, axis=0)
    
    return dy/dt

y = ddt(y)


plt.figure(figsize = (10, 7))
plt.plot(t[0:1500], y)
plt.ylim(-3, 5)
plt.grid()
plt.ylabel('f(t)')
plt.xlabel('t')
plt.title('Plot derivative f\'(t)')
