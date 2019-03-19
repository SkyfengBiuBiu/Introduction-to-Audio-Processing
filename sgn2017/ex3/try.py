import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.signal import argrelextrema
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')
import winsound

class FMsynth(object):
    """
    Simple FM_Synth object for experimentation.
    Defaults: f_carrier = 220, f_mod =220, Ind_mod = 1, length = 5, sampleRate = 44100
    if f_carrier/f_mod = N1/N2 and N1, N2 are integers, harmonic spectra will result
    if N1/N2 is irrational, i.e. sqrt(2) or pi, inharmonic spectra will result 
    f_0, the fundamental = f_carrier/N1 = f_mod/N2
    # k_th harmonic = N1 + n*N2 for n = 0,1,2,3,4,...
    # so for f_carrier = 100 and f_mod = 300, harmonics are [100, 400, 700, 1000, 1300, 1600, 1900, 2200, 2500, 2800] etc
    """
    def __init__(self, f_carrier = 220, f_mod =220, Ind_mod = 1, length = 5, sampleRate = 44100, waveFile = True):
        self.increment = .01
        self.f_carrier = f_carrier
        self.f_mod = f_mod
        self.Ind_mod = Ind_mod
        self.rate = sampleRate
        self.ident = id(self)
        self.name = '%dHz_carrier-%dHz_mod-%s_Index_%d.wav' % (self.f_carrier, self.f_mod, str(self.Ind_mod),self.ident)
        sampleInc = 1.0/self.rate
        x = np.arange(0,length, sampleInc)
        y = np.sin(2*np.pi*self.f_carrier*x + self.Ind_mod*np.sin(2*np.pi*self.f_mod*x))
        mx = 1.059*(max(abs(y))) # scale to max pk of -.5 dB
        y = y/mx
        wavData = np.asarray(32000*y, dtype = np.int16)
        self.wavData = wavData
        if waveFile:
            write('try.wav', 44100, self.wavData)

FM1 = FMsynth() 
Audio(FM1.wavData,rate=FM1.rate)
winsound.PlaySound('try.wav', winsound.SND_FILENAME)