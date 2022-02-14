#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 19:32:30 2022

@author: chadwick
"""
################################################################
# #
# Chadwick Goodall #
# ECE 351 Section 53 #
# Lab 4 #
# 2-15-22 #
################################################################

import causal as cs
import numpy as np
import matplotlib.pyplot as plt

steps = 1e-2
K = 10
t = np.arange(-K, K+steps,steps)
f = 0.25
w = 2*np.pi*f

h1 = np.exp(-2*t) * (cs.u(t) - cs.u(t-3))
h2 = cs.u(t-2) - cs.u(t-6)
h3 = np.cos(w*t) * cs.u(t)

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(t,h1)
#plt.ylim(-3, 5)
plt.grid()
plt.ylabel('h1(t)')
plt.xlabel('')
plt.title('Composite function h1')


plt.subplot(3,1,2)
plt.plot(t,h2)
#plt.ylim(-3, 5)
plt.grid()
plt.ylabel('h2(t)')
plt.xlabel('')
plt.title('Composite function h2')


plt.subplot(3,1,3)
plt.plot(t,h3)
#plt.ylim(-3, 5)
plt.grid()
plt.ylabel('h3(t)')
plt.xlabel('t')
plt.title('Composite function h3')

h1hand = (1/2)*((-1*np.exp(-2*t)+1)* cs.u(t) - (-1*np.exp(-2*(t-3))+1)*cs.u(t-3))
h2hand = ((t-2)*cs.u(t-2) - (t-6)*cs.u(t-6))
h3hand = (1/w)* np.sin(w*t)* cs.u(t)

t = np.arange(-K, K+.5*steps,steps)

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(t,h1hand)
#plt.ylim(-3, 5)
plt.grid()
plt.ylabel('h1hand(t)')
plt.xlabel('')
plt.title('Step response of h1 (hand integral)')


plt.subplot(3,1,2)
plt.plot(t,h2hand)
#plt.ylim(-3, 5)
plt.grid()
plt.ylabel('h2hand(t)')
plt.xlabel('')
plt.title('Step response of h2 (hand integral)')


plt.subplot(3,1,3)
plt.plot(t,h3hand)
#plt.ylim(-3, 5)
plt.grid()
plt.ylabel('h3hand(t)')
plt.xlabel('t')
plt.title('Step response of h3 (hand integral)')

h1step = np.convolve(h1, cs.u(t))
h2step = np.convolve(h2, cs.u(t))
h3step = np.convolve(h3, cs.u(t))

t = np.arange(-2*K, 2*K+.5*steps,steps)

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(t,h1step)
#plt.ylim(-3, 5)
plt.grid()
plt.ylabel('h1step(t)')
plt.xlabel('')
plt.title('Step response of h1 (numpy.convolve)')


plt.subplot(3,1,2)
plt.plot(t,h2step)
#plt.ylim(-3, 5)
plt.grid()
plt.ylabel('h2step(t)')
plt.xlabel('')
plt.title('Step response of h2 (numpy.convolve)')


plt.subplot(3,1,3)
plt.plot(t,h3step)
#plt.ylim(-3, 5)
plt.grid()
plt.ylabel('h3step(t)')
plt.xlabel('t')
plt.title('Step response of h3 (numpy.convolve)')