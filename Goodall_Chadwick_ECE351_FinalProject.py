#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 19:50:09 2022

@author: chadwick
"""

################################################################
# #
# Chadwick Goodall #
# ECE 351 Section 53 #
# Final Project #
# 4-26-22 #
################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack as fftp
import pandas as pd






####################################### FFT function

def MyFft(x,fs):
    N = len(x)
    X_fft = fftp.fft(x)
    X_fft_shifted = fftp.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    
    return freq, X_mag, X_phi

####################################### FFT function with reduced noise

def MyCleanFft(x,fs):
    N = len(x)
    X_fft = fftp.fft(x)
    X_fft_shifted = fftp.fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    
    i = 0
    while (i < len(X_mag)):
        if (X_mag[i] < 1e-10):
            X_phi[i] = 0
        i+=1
    
    return freq, X_mag, X_phi

####################################### Make stem function

def make_stem(ax ,x,y,color='k',style='solid',label='',linewidths =2.5 ,** kwargs):
    ax.axhline(x[0],x[-1],0, color='r')
    ax.vlines(x, 0 ,y, color=color , linestyles=style , label=label , linewidths=linewidths)
    ax.set_ylim ([1.05*y.min(), 1.05*y.max()])


####################################### Task 1
    
fs = 1e6

df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values

# plt.figure(figsize = (10,7))
# plt.plot(t, sensor_sig)
# plt.grid()
# plt.ylabel('Noisy Input Signal')
# plt.xlabel('Time [s]')
# plt.title('Amplitude [V]')
# plt.show()

# freq, X_mag, X_phi = MyCleanFft(sensor_sig, fs)

# fig, ax = plt.subplots(figsize = (10,7))
# make_stem(ax, freq, X_mag)
# plt.xlim(0,1e5)
# plt.show()

# fig, ax = plt.subplots(figsize = (10,7))
# make_stem(ax, freq, X_mag)
# plt.xlim(0,1.8e3)
# plt.show()

# fig, ax = plt.subplots(figsize = (10,7))
# make_stem(ax, freq, X_mag)
# plt.xlim(1.8e3,2e3)
# plt.show()

# fig, ax = plt.subplots(figsize = (10,7))
# make_stem(ax, freq, X_mag)
# plt.xlim(2e3,1e5)
# plt.show()

####################################### Task 3

R = 100
L = .007
C = 1e-6

num = [(1/(R*C)),0]
den =  [1,(1/(R*C)),(1/(L*C))]

steps = 5
w = np.arange(0,1e6+steps*2*np.pi,steps)

lti = sig.lti(num,den)

w, HMag, HPhi = sig.bode(lti, w=w)
w = w/(2*np.pi)


plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.semilogx(w,HMag)
plt.xlim(1,1e5)
plt.grid(True, which='both', ls='-')
plt.ylabel('H magnitude')
plt.title('Magnitude of bandpass filter')


plt.subplot(3,1,2)
plt.semilogx(w,HPhi)
plt.xlim(1,1e5)
plt.grid(True, which='both', ls='-')
plt.ylabel('H phase (deg)')
plt.xlabel('w')
plt.title('Phase of bandpass filter')

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.semilogx(w,HMag)
plt.xlim(1,1.8e3)
plt.grid(True, which='both', ls='-')
plt.ylabel('H magnitude')
plt.title('Magnitude of bandpass filter')

plt.subplot(3,1,2)
plt.semilogx(w,HPhi)
plt.grid(True, which='both', ls='-')
plt.xlim(1,1.8e3)
plt.ylabel('H phase (deg)')
plt.xlabel('w')
plt.title('Phase of bandpass filter')

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.semilogx(w,HMag)
plt.grid(True, which='both', ls='-')
plt.xlim(1.8e3,2e3)
plt.ylim(-1,0)
plt.ylabel('H magnitude')
plt.title('Magnitude of bandpass filter')


plt.subplot(3,1,2)
plt.semilogx(w,HPhi)
plt.grid(True, which='both', ls='-')
plt.xlim(1.8e3,2e3)
plt.ylabel('H phase (deg)')
plt.xlabel('w')
plt.title('Phase of bandpass filter')

plt.figure(figsize = (10, 15))
plt.subplot(3,1,1)
plt.semilogx(w,HMag)
plt.xlim(2e3,100e3)
plt.grid(True, which='both', ls='-')
plt.ylabel('H magnitude')
plt.title('Magnitude of bandpass filter')


plt.subplot(3,1,2)
plt.semilogx(w,HPhi)
plt.grid(True, which='both', ls='-')
plt.xlim(2e3,100e3)
plt.ylabel('H phase (deg)')
plt.xlabel('w')
plt.title('Phase of bandpass filter')

####################################### Task 4

zNum, zDen = sig.bilinear(num,den,fs)

y = sig.lfilter(zNum, zDen, sensor_sig)

plt.figure(figsize = (10, 10))
plt.subplot(3,1,1)
plt.plot(t,y)
plt.grid()
plt.ylabel('Filtered Signal Output')
plt.xlabel('Time [s]')
plt.title('Output of filter with noisy signal input (Amplitude [V])')

freq, X_mag, X_phi = MyCleanFft(y, fs)

fig, ax = plt.subplots(figsize = (10,7))
make_stem(ax, freq, X_mag)
plt.xlim(0,1e5)
plt.show()

fig, ax = plt.subplots(figsize = (10,7))
make_stem(ax, freq, X_mag)
plt.xlim(0,1.8e3)
plt.show()

fig, ax = plt.subplots(figsize = (10,7))
make_stem(ax, freq, X_mag)
plt.xlim(1.8e3,2e3)
plt.show()

fig, ax = plt.subplots(figsize = (10,7))
make_stem(ax, freq, X_mag)
plt.xlim(2e3,1e5)
plt.show()