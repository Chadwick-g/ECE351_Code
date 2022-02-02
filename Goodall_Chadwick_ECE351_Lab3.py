#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################
# #
# Chadwick Goodall #
# ECE 351 Section 53 #
# Lab 3 #
# 2-8-22 #
################################################################

import numpy as np
import matplotlib.pyplot as plt
#import scipy.signal as sig


steps = 1e-2
t = np.arange(0, 20+steps,steps) 

# beginning of the step function and plot

def u(t):
    y = np.zeros(t.shape)

    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = 1
            
    return y    

# beginning of the ramp function and plot
  
def r(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            y[i] = 0
        else:
            y[i] = t[i]
        
    return y

f1 = u(t-2)-u(t-9)
f2 = np.exp(-t) * u(t)
f3 = (r(t-2) * (u(t-2)-u(t-3))) + (r(4-t) * (u(t-3)-u(t-4)))

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(t,f1)
#plt.ylim(-3, 5)
plt.grid()
plt.ylabel('f1(t)')
plt.xlabel('t')
plt.title('Composite function f1')

plt.subplot(3,1,2)
plt.plot(t,f2)
#plt.ylim(-3, 5)
plt.grid()
plt.ylabel('f2(t)')
plt.xlabel('t')
plt.title('Composite function f2')

plt.subplot(3,1,3)
plt.plot(t,f3)
plt.grid()
plt.ylabel('f3(t)')
plt.xlabel('t')
plt.title('Composite function f3')

def conv(f1,f2):
    f1len = len(f1)
    f2len = len(f2)
    
    #extend arrays so they are same size
    f1Ext = np.append(f1, np.zeros((1,f2len-1)))
    f2Ext = np.append(f2, np.zeros((1,f1len-1)))
    
    # init new array that is of same size to contain the result
    res = np.zeros(f1Ext.shape)
    
    for i in range(f1len+f2len-2):
        res[i]=0
        for j in range(f1len):
            if (i-j+1 > 0):
                try:
                    res[i] += f1Ext[j] * f2Ext[i-j+1]
                except:
                    print(i,j)
    return res
            
            
f1Convf2 = conv(f1,f2)
f2Convf3 = conv(f2,f3)
f1Convf3 = conv(f1,f3)

npf1Convf2 = np.convolve(f1, f2)
npf2Convf3 = np.convolve(f2,f3)
npf1Convf3 = np.convolve(f1,f3)

t = np.arange(0, 40+steps*3,steps) 

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(t,f1Convf2)
plt.grid()
plt.ylabel('f1(t)*f2(t)')
plt.xlabel('t')
plt.title('f1 convovled with f2')

plt.subplot(3,1,2)
plt.plot(t,f2Convf3)
plt.grid()
plt.ylabel('f2(t)*f3(t)')
plt.xlabel('t')
plt.title('f2 convovled with f3')

plt.subplot(3,1,3)
plt.plot(t,f1Convf3)
plt.grid()
plt.ylabel('f1(t)*f3(t)')
plt.xlabel('t')
plt.title('f1 convovled with f3')

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(t,npf1Convf2)
plt.grid()
plt.ylabel('f1(t)*f2(t)')
plt.xlabel('t')
plt.title('f1 convovled with f2 (numpy.convolve)')

plt.subplot(3,1,2)
plt.plot(t,npf2Convf3)
plt.grid()
plt.ylabel('f2(t)*f3(t)')
plt.xlabel('t')
plt.title('f2 convovled with f3 (numpy.convolve)')

plt.subplot(3,1,3)
plt.plot(t,npf1Convf3)
plt.grid()
plt.ylabel('f1(t)*f3(t)')
plt.xlabel('t')
plt.title('f1 convovled with f3 (numpy.convolve)')
    
    