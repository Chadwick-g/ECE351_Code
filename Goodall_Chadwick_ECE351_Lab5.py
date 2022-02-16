#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 19:26:01 2022

@author: chadwick
"""

################################################################
# #
# Chadwick Goodall #
# ECE 351 Section 53 #
# Lab 5 #
# 2-22-22 #
################################################################

#import causal as cs
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

################## Part 1

steps = 1e-6
t = np.arange(0, 0.0012+steps,steps)

num = [0, .027, 0]
den = [1000*.027*(10**-7), .027, 1000]
h_t = 10000*np.exp(-5000*t)*(np.cos(18584*t)-0.269*np.sin(18584*t))
tout, yout = sig.impulse((num,den), T=t)

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(t,h_t)
plt.grid()
plt.ylabel('h(t)')
plt.xlabel('')
plt.title('Impulse response')


plt.subplot(3,1,2)
plt.plot(tout,yout)
plt.grid()
plt.ylabel('h(t)')
plt.xlabel('t')
plt.title('Impulse response from sig.impulse')

################## Part 2

lti = sig.lti(num,den)
tout, yout = sig.step(lti)

plt.figure(figsize = (10, 10))
#plt.subplot(3,1,1)
plt.plot(tout,yout)
plt.grid()
plt.ylabel('h(t)')
plt.xlabel('t')
plt.title('Step response')
