#!/usr/bin/env python3
# Lien github 
# 
# -*- coding: utf-8 -*-
#
#   shift(register, feedback, output)
#
#

"""
Created on Tue Sep 29 19:31:22 2020

"""

import matplotlib.pyplot as plt
#import mplleaflet
import numpy as np
import scipy.signal as sig

from scipy import signal


def shift(register, feedback, output):
    """GPS Shift Register

    :param list feedback: which positions to use as feedback (1 indexed)
    :param list output: which positions are output (1 indexed)
    :returns output of shift register:

    """

    # calculate output
    out = [register[i-1] for i in output]
    if len(out) > 1:
        out = sum(out) % 2
    else:
        out = out[0]

    # modulo 2 add feedback
    fb = sum([register[i-1] for i in feedback]) % 2

    # shift to the right
    for i in reversed(range(len(register[1:]))):
        register[i+1] = register[i]

    # put feedback in position 1
    register[0] = fb

    return out


def PRN(sv):
    """Build the CA code (PRN) for a given satellite ID

    :param int sv: satellite code (1-32)
    :returns list: ca code for chosen satellite

    """
    SV = {
   1: [2,6],
   2: [3,7],
   3: [4,8],
   4: [5,9],
   5: [1,9],
   6: [2,10],
   7: [1,8],
   8: [2,9],
   9: [3,10],
  10: [2,3],
  11: [3,4],
  12: [5,6],
  13: [6,7],
  14: [7,8],
  15: [8,9],
  16: [9,10],
  17: [1,4],
  18: [2,5],
  19: [3,6],
  20: [4,7],
  21: [5,8],
  22: [6,9],
  23: [1,3],
  24: [4,6],
  25: [5,7],
  26: [6,8],
  27: [7,9],
  28: [8,10],
  29: [1,6],
  30: [2,7],
  31: [3,8],
  32: [4,9] }
    # init registers
    G1 = [1 for i in range(10)]
    G2 = [1 for i in range(10)]

    ca = [] # stuff output in here

    # create sequence
    for i in range(1023):
        g1 = shift(G1, [3,10], [10])
        g2 = shift(G2, [2,3,6,8,9,10], SV[sv]) # <- sat chosen here from table

        # modulo 2 add and append to the code
        ca.append((g1 + g2) % 2)
    ca = [-1 if x==0 else x for x in ca]
    # return C/A code!
    return ca


def detect_start(r, T,  a):
    'guard interval correlation'
    tomax= signal.correlate(r, a, 'valid')   # correlation
    ind,  peaks = signal.find_peaks(np.abs(tomax) , distance=  0.9*  T ,  height= 0.1*np.max(np.abs(tomax)))

    # rr= np.copy(r[ind[0]: ind[0]+ 10*T])
    tosumd = np.conj(r[ :-T]) * r[ T: ] #matrix of complex conjugate multiplication to be summed

    tomaxd  =  np.convolve( tosumd,np.ones( T ) ,'valid')  #sum of window length T

    # see= np.zeros(len(tomax))
    # a= np.abs(tomax)

    # see[ ind  ]= 1
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(np.abs(tomax) )
    # plt.subplot(212)
    # plt.plot(see)
    return tomax,tomaxd, ind  ; #tomax is the correlation function



def correlate(rc,csat,fd):
    """
    rc : complex signal IQ samples
    csat : code pseudo aléatoire échantillonné à fs
    """
    t = np.arange(0,len(rc) )
    rr = rc[None,:]*np.exp(-1j*2*np.pi*t*fd[:,None])
    RR = np.fft.fft(rr)
    CSAT = np.fft.fft(csat)
    U = np.fft.ifft(RR* np.conj(CSAT[None,:]))
    return(U)

def codesat(isat,repeatno=200,fs=2600000,fsca=1.023e6):
    """ Construit un motif du code PRN du satellite à la bonne fréquence d'ech

    Parameters
    ----------
    isat : satellite number
    repeatno : int number of code repetition
    fs   : sampling frequency
    fsca : frequence code C/A
    Returns
    -------
    csat : repetion of C/A code (sampled @ fs)

    """
    sat = np.array( PRN(isat) ) # code PRN du satellite
    satind  =  np.floor( np.linspace(0, len(sat)-1, num=int((fs/fsca)* len(sat)) )     ).astype(int)
    satrep  = sat[satind]
    csat = np.tile(satrep,repeatno)
    return(csat)

# 
def identify_satellites(signal,fd,repeatno=20):
    """
    Fonction pour identifier les satellites présents dans le signal
    

    Parameters
    ----------
    signal : np.array 
        complex signal 
    fd : np.array
        Doppler range 
    repeatno : int, optional
        number of repetition of the code. The default is 20.

    Returns
    -------
    detected_satellites : TYPE
        DESCRIPTION.
    correlations : TYPE
        DESCRIPTION.
    largmax : TYPE
        DESCRIPTION.

    """
    # Corrélation croisée entre le signal et les codes C/A des satellites connus
    correlations = []
    largmax = []
    for sat_num in range(1,32):
        csat = codesat(sat_num,repeatno=repeatno)
        rc = signal[0:len(csat)]
        U = correlate(rc,csat,fd)
        corrmax = np.max(np.abs(U))
        a = np.where(np.abs(U)==corrmax)
        correlations.append(corrmax)
        largmax.append(a)
        #correlation = np.abs(fftconvolve(signal, code[::-1], mode='same'))
    # Identifier les pics de corrélation pour déterminer les satellites présents
    threshold = 0.5 * np.max(correlations)
    detected_satellites = [i+1 for i, corr in enumerate(correlations) if corr > threshold]
    largmax = [ largmax[i-1] for i in detected_satellites ]
    largmax = [ (a[0][0],a[1][0]) for a in largmax ]
    return detected_satellites,correlations,largmax

def fine_Doppler(signal,detected_satellites,fd,larg,Nfd=100,kstart=0,repeatno=20):
    correlations = []
    lfineDoppler =[]
    for k,sat_num in enumerate(detected_satellites):
        csat = codesat(sat_num,repeatno=repeatno)
        idxfdsat = larg[k][0]
        fdsat  = np.linspace(fd[idxfdsat-1],fd[idxfdsat+1],Nfd)
        rc = signal[kstart:len(csat)+kstart]
        U = correlate(rc,csat,fdsat)
        corrmax = np.max(np.abs(U))
        a = np.where(np.abs(U)==corrmax)
        correlations.append(corrmax)
        lfineDoppler.append(fdsat[a[0][0]])
    return lfineDoppler,correlations
