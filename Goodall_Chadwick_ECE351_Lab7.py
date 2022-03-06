#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:30:38 2022

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
import scipy.signal as sig
import matplotlib.pyplot as plt

steps = 1e-3
t = np.arange(0, 5+steps,steps)

################## Part 1

gNum = [1,9]
gDen = [1,-2,-40,-64]
aNum = [1,4]
aDen = [1,4,3]
b = [1,26,168]

gz, gp, gk = sig.tf2zpk(gNum, gDen)
az, ap, ak = sig.tf2zpk(aNum, aDen)
bOut = np.roots(b)

print ("zeros of g: " + str(gz) + " poles of g: " + str(gp) + " gain of g: " + str(gk))
print ("zeros of a: " + str(az) + " poles of a: " + str(ap) + " gain of a: " + str(ak))
print ("roots of b: " + str(bOut))

num = sig.convolve([1,9],[1,4])
den = sig.convolve(gDen,aDen)

lti = sig.lti(num,den)
tout, yout = sig.step(lti)

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(tout,yout)
plt.grid()
plt.ylabel('Y(t)')
plt.xlabel('t')
plt.title('Open loop step response')

################## Part 2
# closed loop zeros, poles, and gain
CLNum = sig.convolve(gNum,aNum)
CLDen1 = sig.convolve(gDen,aDen)
CLDen2 = sig.convolve(b,sig.convolve(gNum,aDen))
CLDen =  CLDen1+ CLDen2
#CLDen = sig.convolve(aDen,(gDen + sig.convolve(b, gNum)))
CLz, Clp, Clk = sig.tf2zpk(CLNum, CLDen)

lti = sig.lti(CLNum,CLDen)
tout, yout = sig.step(lti)

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.plot(tout,yout)
plt.grid()
plt.ylabel('Y(t)')
plt.xlabel('t')
plt.title('Closed loop step response')
