#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 19:29:48 2022

@author: chadwick
"""

################################################################
# #
# Chadwick Goodall #
# ECE 351 Section 53 #
# Lab 5 #
# 2-22-22 #
################################################################

import causal as cs
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def cos_pfd(r, p, t):
    y = 0
    for i in range (0, len(r)):
        k_mag = np.absolute(r[i])
        k_angle = np.angle(r[i])
        a = np.real(p[i])
        w = np.imag(p[i])
        y += k_mag*np.exp(a*t)*np.cos(w*t + k_angle) * cs.u(t)    
    return y


steps = 1e-3
t = np.arange(0, 2+steps,steps)


#h_t = 10000*np.exp(-5000*t)*(np.cos(18584*t)-0.269*np.sin(18584*t))
#tout, yout = sig.impulse((num,den), T=t)

################## Part 1

y_t = (0.5 - 0.5*np.exp(-4*t) + np.exp(-6*t))*cs.u(t)

num = [1, 6, 12]
den = [1, 10, 24]

lti = sig.lti(num,den)
tout, yout = sig.step(lti)

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(t,y_t)
plt.grid()
plt.ylabel('Y(t)')
plt.xlabel('')
plt.title('Y(t) Hand Calculation')


plt.subplot(3,1,2)
plt.plot(tout,yout)
plt.grid()
plt.ylabel('Y(t)')
plt.xlabel('t')
plt.title('Y(t) scipy.signal.step Calculation')

num = [0, 1, 6, 12] 
den = [1, 10, 24, 0]

r, p, k = sig.residue(num, den)

print("residues: " + str(r) + "\npoles: " + str(p) + "\ncoefficients: " + str(k))

################## Part 2

#task 1
t = np.arange(0, 4.5+steps,steps)

num = [0, 0, 0, 0, 0, 0, 25250]
den = [1, 18, 218, 2036, 9085, 25250, 0]
r, p, k = sig.residue(num, den)
print("\nresidues: " + str(r) + "\npoles: " + str(p) + "\ncoefficients: " + str(k))

#task 2
y_t = cos_pfd(r, p, t)

#task 3
num = [25250]
den = [1, 18, 218, 2036, 9085, 25250]

#find y_t using H_s
tout, yout = sig.step((num,den), T = t)

plt.figure(figsize = (10, 10))
plt.subplot(3,1,1)
plt.plot(t,y_t)
plt.grid()
plt.ylabel('Y(t)')
plt.xlabel('')
plt.title('Y(t) using custom cosine function')


plt.subplot(3,1,2)
plt.plot(tout,yout)
plt.grid()
plt.ylabel('Y(t)')
plt.xlabel('t')
plt.title('Y(t) using step function')



