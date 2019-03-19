# coding: utf-8

# import Python modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from scipy import signal
import peakutils

# to install this package, you need to use the command line tool "pip"
# pip install peakutils --user 

## Edit the script to synthesize audio by means of Frequency Modulation (FM)

# load the audio files and resample (decimate by a factor 4)
# fs_org, x_org = wavfile.read('gtr55.wav')
fs_org, x_org = wavfile.read('oboe59.wav')
decimation_factor = 4
x = scipy.signal.decimate(x_org, decimation_factor, ftype='iir')
fs = int(fs_org / decimation_factor)
t_scale = np.linspace(0,len(x)/fs,num=len(x))

# Time domain visualization
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(t_scale, x)
plt.title('Time domain visualization of the audio signal')
plt.grid('on')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Apply DFT : In this exercice, we're going to use windowed DFT. This means that we apply a window to the signal before
# FFT in order to "smooth" it at the boundaries. Therefore, the procedure is:
# 1/ Take a slice x_slice of x (as in exercice 1)(e.g., x_slice = x[t_start:t_start+win_len])
# 2/ Multiply it with a window: x_win = win * x_slice
# 3/ Get the fft of x_win

# Define the window
win_len = 8192
win = scipy.signal.get_window('hann', win_len)

# apply windowed DFT:
t_start = 10000
X = scipy.fft(x[t_start:t_start+win_len]*win)
mag_X = 20 * np.log10(np.abs(X[0:int(win_len / 2) - 1]))
freq_scale = np.linspace(0, fs / 2.0, int(win_len / 2) - 1)

# Frequency domain visualization
plt.subplot(2, 1, 2)
plt.plot(freq_scale, mag_X)
plt.title('Frequency domain visualization of the audio signal')
plt.grid('on')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.tight_layout()

# Find peaks.
peak_indexes = peakutils.indexes(mag_X, thres=0.9, min_dist=np.round(100 / fs * win_len))

# Convert the peak indexes into frequencies (in Hz)
peak_freqs = freq_scale[peak_indexes]

# Only keep indexes which correspond to frequencies above 20Hz (threshold of human hearing)
# since there might be extra noise peaks below
peak_indexes = peak_indexes[peak_freqs > 20]

# Get the frequencies and magnitudes
peak_freqs = freq_scale[peak_indexes]
peak_amps = 20 * np.log10(np.abs(peak_freqs))

# Plot the peak locations and magnitudes.
plt.plot(peak_freqs, peak_amps, 'o')
plt.show()

# Q1: Save the plot and include it in your report. The plot should have the following information: (0.2 pt)
# - time domain visualization
# - frequency domain visualization
# - marker at the peaks in the frequency domain

# Q2: Are the peaks detected using thres=0.7 correct?
# Change the value of the threshold (between 0 and 1) and describe what happens. (0.2 pt)?
peaks_varying=[]
for thre_i in np.linspace(0.0,1.0,10):
    peak_indexes_changed = peakutils.indexes(mag_X, thres=thre_i, min_dist=np.round(100 / fs * win_len))
    peak_freqs_changed_means = np.mean(freq_scale[peak_indexes_changed])
    peaks_varying.append(peak_freqs_changed_means)
plt.figure()
plt.plot(np.linspace(0.0,1.0,10), peaks_varying)
plt.title('Frequency varying by the threshold')
plt.grid('on')
plt.ylabel('Mean peak frequency')
plt.xlabel('Threshold')
plt.tight_layout()

# Q3: What is the fundamental frequency? Calculate the ratios peak_freqs/F0 and copy them in your report (0.2 pt)
F0 = 246.4
print(F0)
ratios = peak_freqs/F0
print(ratios)

# Q4: Is the oboe sound (oboe59) harmonic? Why/why not? (0.2 pt)
#Yes, if many vibration modes are typically nearly exact integer multiples of a fundamental frequency, then this mode would be harmonic.
#In this way, the ratios which are similar to interger and the oboe sound is harmonic.

# Q5: Same for the other recording (gtr55): Plot the time/frequency domain representations with peak markers,
# compute the fundamental frequency and the ratio Fn/F0, and conclude about harmonicity. (0.2 pt)


# Simple FM Synthesis
# The FM synthesis formula is:
# y = A * sin( 2*pi*f_carrier + ind_mod*sin(2*pi*f_mod))
# with:
# A: amplitude
# f_carrier: a carrier frequency
# f_mod: a modulation frequency
# ind_mod: a modulation index, which interact to generate harmonic and non-harmonic sounds

# Implement FM synthesis with modulation index of 9
F1 = peak_freqs[1]
ind_mod = 20
f_carrier = F0
f_mod = F1 - F0
A = 10 ** (peak_amps[0] / 20)

y = None

# Perform FFT: remember to apply a hann window to the signal before FFT (as in the first part)
Y = None
mag_Y = 20 * np.log10(np.abs(Y[0:int(win_len / 2.0) - 1]))

# Q6: Plot with time/frequency domain visualization of y (you can reuse some code from the first part)
# and include it in your report (0.4 pt)?

# Record the audio file
audio_name = 'fm_carrier-%dHz_mod-%s.wav' % (f_carrier, str(ind_mod))
wavfile.write(audio_name, fs, y / np.max(np.abs(y)))

# Q7: Does the FM synthesis version sound similar or different from the original signal? (0.3 pt)
# Is it better, in your opinion, than the simple sinusoid synthesis from exercice 1?
# How could you improve it?

# Q8: Try different values for the modulation index (e.g., 0, 20, 100).
# Look at the spectrum and listen at the sound. Describe how it affects the sound (0.3 pt)