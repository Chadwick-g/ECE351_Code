#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 19:28:57 2022

@author: chadwick
"""

################################################################
# #
# Chadwick Goodall #
# ECE 351 Section 53 #
# Lab 10 #
# 4-5-22 #
################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con

steps = 1000
R = 1e3
L = 27e-3
C = 100e-9
w = np.arange(1e3,1e6+steps,steps)


####################################### Part 1 ################################

####################################### Task 1

HMag = 20*np.log10(((1/(R*C))*w)/(np.sqrt(w**4+((1/(R*C))**2 - (2/(L*C)))*w**2 + (1/(L*C))**2)))
HPhi = (180/np.pi)*(np.pi/2 - np.arctan(((1/(R*C))*w)/(-w**2+ (1/(L*C)))))

i = 0
while (i < len(w)):
    if (HPhi[i] > 90):
        HPhi[i] -= 180
    i += 1

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.semilogx(w,HMag)
plt.grid()
plt.ylabel('H magnitude')
plt.title('Magnitude of transfer function')


plt.subplot(3,1,2)
plt.semilogx(w,HPhi)
plt.grid()
plt.ylabel('H phase')
plt.xlabel('w')
plt.title('Phase of transfer function')

####################################### Task 2

# numerator and denominator needed for lti conversion
num = [(1/(R*C)),0]
den =  [1,(1/(R*C)),(1/(L*C))]

# lti conversion necessary for input to sig.bode
lti = sig.lti(num,den)

w, HMag, HPhi = sig.bode(lti, w=w)

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.semilogx(w,HMag)
plt.grid()
plt.ylabel('H magnitude')
plt.title('Magnitude of transfer function (sig.bode)')


plt.subplot(3,1,2)
plt.semilogx(w,HPhi)
plt.grid()
plt.ylabel('H phase')
plt.xlabel('w')
plt.title('Phase of transfer function (sig.bode)')

####################################### Task 3

plt.figure()
sys = con.TransferFunction(num, den)
_ = con.bode(sys, w, dB = True, Hz = True, deg = True, plot = True)

####################################### Part 2 ################################

####################################### Task 1

fs = 2*np.pi*50e3
steps = 1/fs
t = np.arange(0,1e-2+steps,steps)
x = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50e3*t)

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid()
plt.ylabel('x(t)')
plt.xlabel('t')
plt.title('X(t) function (Part 2)')

####################################### Task 2

fs =  2*np.pi*100
zNum, zDen = sig.bilinear(num,den,fs)

####################################### Task 3

y = sig.lfilter(zNum, zDen, x)

####################################### Task 4

plt.figure(figsize = (10, 10))
plt.subplot(3,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel('y(t)')
plt.xlabel('t')
plt.title('Y(t) function (Part 2 task 4)')

