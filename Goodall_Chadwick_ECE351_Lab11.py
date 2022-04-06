#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:52:32 2022

@author: chadwick
"""

################################################################
# #
# Chadwick Goodall #
# ECE 351 Section 53 #
# Lab 11 #
# 4-12-22 #
################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import scipy.signal as sig

def zplane(b, a, filename = None):
        """ Plot the complex zplane given a transfer function"""
        
        # get plot
        ax = plt.subplot (1,1,1) 
        
        # create unit circle
        uc = patches.Circle((0,0),radius = 1,fill = False,color = 'black', ls = 'dashed' )
        ax.add_patch(uc)
        
        #if coeffs are less than 1, normalize
        if np.max(b) >1:
                kn = np.max(b)
                b = np.array(b)/float(kn)
        else :
            kn = 1
            
        if np.max(a) >1:
                kd = np.max(a)
                a = np.array(a)/float(kd)
        else :
            kd = 1
        
        # get poles and zeros
        p = np.roots(a)
        z = np.roots(b)
        k = kn/float(kd)
        
        #plot the zeros and set marker properties
        t1 = plt.plot(z.real , z.imag , 'o', ms=10,label='Zeros')
        plt.setp(t1 , markersize =10.0, markeredgewidth =1.0)
        
        # plot the poles and set marker properties
        t2 = plt.plot(p.real , p.imag , 'x', ms=10,label='Poles')
        plt.setp( t2 , markersize =12.0, markeredgewidth =3.0)
        
        ax.spines['left']. set_position('center')
        ax.spines['bottom']. set_position('center')
        ax.spines['right']. set_visible(False)
        ax.spines['top']. set_visible(False)
        
        plt.legend ()
        # set the ticks

        # r = 1.5; plt.axis('scaled '); plt.axis([-r, r, -r, r])
        # ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)
        
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            
        return z, p, k
        
    
H_znum = [2, -40]
H_zden = [1, -10, 16] 
z, p, k = sig.residuez(H_znum, H_zden)
print('residuez residues: ' + str(z) + ' poles: ' + str(p) + ' coeffs: ' + str(k))
z, p, k = zplane(H_znum, H_zden)
print('zplane residues: ' + str(z) + ' poles: ' + str(p) + ' coeffs: ' + str(k))
w, h = sig.freqz(H_znum, H_zden, whole=True)

plt.figure(figsize = (10, 18))
plt.subplot(3,1,1)
plt.plot(w,20*np.log10(np.abs(h)))
plt.grid()
plt.ylabel('Magnitude |Hdb|')
plt.xlabel('Frequency w')
plt.title('Magnitude and Phase plots')

plt.subplot(3,1,2)
plt.plot(w,np.angle(h, deg=True))
plt.grid()
plt.ylabel('Phase angle')
plt.xlabel('Frequency w')