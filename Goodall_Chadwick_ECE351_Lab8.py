#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 19:34:11 2022

@author: chadwick
"""

################################################################
# #
# Chadwick Goodall #
# ECE 351 Section 53 #
# Lab 7 #
# 2-22-22 #
################################################################

import numpy as np
#import scipy.signal as sig
import matplotlib.pyplot as plt

steps = 1e-2
t = np.arange(0, 20+steps,steps)
T = 8
N = [1,3,15,50,150,1500]
w0 = (2*np.pi)/T 

print('a0 = a1 = 0')
b = np.zeros(4)
for i in range(1,4):
    b[i] = (2/(i*np.pi))*(1-np.cos(i*np.pi))
    print(str(b[i]))
    
   

# for i in range(0,len(N)+1): # for each value in array N
#     n = N[i] # set n equal to a val in N
#     b = np.zeros(n) # zero an array of size N[i] 
#     currSum = np.zeros(n)
#     for k in range(1,n+1):
#         b[k] = (2/(k*np.pi))*(1-np.cos(k*np.pi))
#         currSum += b[k]*np.sin(k*w0*t)
#     xt = currSum
    
#b = np.zeros(N[0])
#currSum = np.zeros(N[0])

# for k in np.arange(1,N[0]):
#     b = (2/(k*np.pi))*(1-np.cos(k*np.pi))
#     currSum = b*np.sin(k*w0*t)
#     #print(str(currSum))
#     xt += currSum

def x(t, T, N):
    k = 1
    w0 = (2*np.pi)/T 
    x = np.zeros(t.shape)
    while (k <= N):
            x += (2/(k*np.pi))*(1-np.cos(k*np.pi))*np.sin(k*w0*t)
            k += 1
    return x

x1 = x(t, T, N[0])
x2 = x(t, T, N[1])
x3 = x(t, T, N[2])
x4 = x(t, T, N[3])
x5 = x(t, T, N[4])
x6 = x(t, T, N[5])

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(t,x1)
plt.grid()
plt.ylabel('X(t)')
plt.xlabel('t')
plt.title('X(t)')

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(t,x2)
plt.grid()
plt.ylabel('X(t)')
plt.xlabel('t')
plt.title('X(t)')

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(t,x3)
plt.grid()
plt.ylabel('X(t)')
plt.xlabel('t')
plt.title('X(t)')

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(t,x4)
plt.grid()
plt.ylabel('X(t)')
plt.xlabel('t')
plt.title('X(t)')

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(t,x5)
plt.grid()
plt.ylabel('X(t)')
plt.xlabel('t')
plt.title('X(t)')

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(t,x6)
plt.grid()
plt.ylabel('X(t)')
plt.xlabel('t')
plt.title('X(t)')