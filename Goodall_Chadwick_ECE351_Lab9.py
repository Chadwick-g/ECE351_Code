#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 19:32:37 2022

@author: chadwick
"""

################################################################
# #
# Chadwick Goodall #
# ECE 351 Section 53 #
# Lab 8 #
# 3-29-22 #
################################################################

import numpy as np
import scipy.fftpack as fftp
import matplotlib.pyplot as plt

fs = 10
steps = 1/fs
t = np.arange(0, 2,steps)
x = np.cos(2*np.pi*t)
T = 8
N = 15

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

####################################### Series approximation function

def fSeries(t, T, N):
    k = 1
    w0 = (2*np.pi)/T 
    x = np.zeros(t.shape)
    while (k <= N):
            x += (2/(k*np.pi))*(1-np.cos(k*np.pi))*np.sin(k*w0*t)
            k += 1
    return x

# for reference
# plt.figure(figsize = (10, 15))
# plt.subplot(3,1,1)
# plt.plot(t,x1)
# plt.grid()
# plt.ylabel('X(t)')
# plt.xlabel('t')
# plt.title('X(t)')

####################################### Task 1

freq, X_mag, X_phi = MyFft(x, fs)

plt.figure(figsize = (15, 15))
plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid()
plt.ylabel('x(t)')
plt.xlabel('t')
plt.title('User Defined FFT of Cos(pi*2*t)')


plt.subplot(3,2,3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('|x(f)|')

plt.subplot(3,2,5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,4)
plt.stem(freq, X_mag)
plt.xlim(-2,2)
plt.grid()
plt.ylabel('|x(f)|')

plt.subplot(3,2,6)
plt.stem(freq, X_phi)
plt.xlim(-2,2)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

####################################### Task 2

x = 5*np.sin(2*np.pi*t)
freq, X_mag, X_phi = MyFft(x, fs)

plt.figure(figsize = (15, 15))
plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid()
plt.ylabel('x(t)')
plt.xlabel('t')
plt.title('User Defined FFT of x(t) task 2')


plt.subplot(3,2,3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('|x(f)|')

plt.subplot(3,2,5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,4)
plt.stem(freq, X_mag)
plt.xlim(-2,2)
plt.grid()
plt.ylabel('|x(f)|')

plt.subplot(3,2,6)
plt.stem(freq, X_phi)
plt.xlim(-2,2)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

####################################### Task 3

x = (2*np.cos(((2*np.pi*2*t)-2))) + (np.sin(((2*np.pi*6*t)+3))**2)
freq, X_mag, X_phi = MyFft(x, fs)

plt.figure(figsize = (15, 15))
plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid()
plt.ylabel('x(t)')
plt.xlabel('t')
plt.title('User Defined FFT of x(t) task 3')


plt.subplot(3,2,3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('|x(f)|')

plt.subplot(3,2,5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,4)
plt.stem(freq, X_mag)
plt.xlim(-15,15)
plt.grid()
plt.ylabel('|x(f)|')

plt.subplot(3,2,6)
plt.stem(freq, X_phi)
plt.xlim(-15,15)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

####################################### Task 4

x = np.cos(2*np.pi*t)
freq, X_mag, X_phi = MyCleanFft(x, fs)

plt.figure(figsize = (15, 15))
plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid()
plt.ylabel('x(t)')
plt.xlabel('t')
plt.title('User Defined Clean FFT of Cos(pi*2*t)')


plt.subplot(3,2,3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('|x(f)|')

plt.subplot(3,2,5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,4)
plt.stem(freq, X_mag)
plt.xlim(-2,2)
plt.grid()
plt.ylabel('|x(f)|')

plt.subplot(3,2,6)
plt.stem(freq, X_phi)
plt.xlim(-2,2)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

x = 5*np.sin(2*np.pi*t)
freq, X_mag, X_phi = MyCleanFft(x, fs)

plt.figure(figsize = (15, 15))
plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid()
plt.ylabel('x(t)')
plt.xlabel('t')
plt.title('User Defined Clean FFT of x(t) task 2')


plt.subplot(3,2,3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('|x(f)|')

plt.subplot(3,2,5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,4)
plt.stem(freq, X_mag)
plt.xlim(-2,2)
plt.grid()
plt.ylabel('|x(f)|')

plt.subplot(3,2,6)
plt.stem(freq, X_phi)
plt.xlim(-2,2)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

x = (2*np.cos(((2*np.pi*2*t)-2))) + (np.sin(((2*np.pi*6*t)+3))**2)
freq, X_mag, X_phi = MyCleanFft(x, fs)

plt.figure(figsize = (15, 15))
plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid()
plt.ylabel('x(t)')
plt.xlabel('t')
plt.title('User Defined Clean FFT of x(t) task 3')


plt.subplot(3,2,3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('|x(f)|')

plt.subplot(3,2,5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,4)
plt.stem(freq, X_mag)
plt.xlim(-15,15)
plt.grid()
plt.ylabel('|x(f)|')

plt.subplot(3,2,6)
plt.stem(freq, X_phi)
plt.xlim(-15,15)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

####################################### Task 5

t = np.arange(0, 16,steps)
x = fSeries(t, T, N)
freq, X_mag, X_phi = MyCleanFft(x, fs)

plt.figure(figsize = (15, 15))
plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid()
plt.ylabel('x(t)')
plt.xlabel('t')
plt.title('User Defined Clean FFT of x(t) task 5')


plt.subplot(3,2,3)
plt.stem(freq, X_mag)
plt.grid()
plt.ylabel('|x(f)|')

plt.subplot(3,2,5)
plt.stem(freq, X_phi)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,4)
plt.stem(freq, X_mag)
plt.xlim(-2,2)
plt.grid()
plt.ylabel('|x(f)|')

plt.subplot(3,2,6)
plt.stem(freq, X_phi)
plt.xlim(-2,2)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

