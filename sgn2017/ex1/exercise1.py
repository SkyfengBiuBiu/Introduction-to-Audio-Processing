# coding: utf-8

# import Python modules
import winsound
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from scipy import signal

## 1.
# Edit the script to analyse the audio wav file (short recording of a musical instrument) in time domain and frequency domain using python modules
# you can resuse parts of exercise0.py from your previous exercise
#!! FILL IN PARTS WITH "None"

# Load audio signal
fs, x = wavfile.read('glockenspiel-a-2.wav')
number_of_samples = len(x)
duration = len(x)/fs
print('{} {}'.format('Sampling frequency (in Hz) is: ', fs))
print('{} {}'.format('Duration (in s) is: ', duration))

t_scale = np.linspace(0,len(x)/fs,num=len(x)) # Hint: use the function np.linspace
amp_scale = x

# Plot time domain visualization
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(t_scale, amp_scale)
plt.title('Time domain visualization of the audio signal')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')

# Plot frequency domain visualization using FFT
max_freq = fs/2
win_len = 2048
number_frequencies = int(win_len / 2) - 1
t_start = 10000
X = scipy.fft(x[t_start:t_start+win_len])
frq_scale = np.linspace(0, max_freq, int(win_len/2)-1)
mag_scale = 20*np.log10(np.abs(X[0:int(win_len / 2) - 1]))

plt.subplot(2, 1, 2)
plt.plot(frq_scale, mag_scale)
plt.title('Frequency domain visualization of the audio signal')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

## 2
# Edit the script to generate a Sine wave signal y0,
# with amplitude 3000 (so you will hear something if you play the signal),
# frequency F0, and length of t seconds, as calculated from step 1.

# Generate sine wave y0
F0 = 1877 # To be checked from the previous plot
amplitude = 3000
y0 = amplitude* np.sin(2*np.pi*F0*t_scale)

# Plot time domain visualization
short_length = 100
t_scale_short = np.linspace(0,len(x)/fs,num=short_length)
amp_scale_short = y0[0:short_length]

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(t_scale_short, amp_scale_short)
plt.title('Time domain visualization of the sine wave y0')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time (samples)')

# Apply fast Fourier transform (FFT) to the Sine wave. Display the FFT of the waveform.

# Plot frequency domain visualization using FFT
Y0 = scipy.fft(y0[t_start:t_start+win_len])
mag_scale = 20*np.log10((Y0[0:int(win_len / 2) - 1]))

plt.subplot(2, 1, 2)
plt.plot(frq_scale, mag_scale)
plt.title('Frequency domain visualization of the sine wave y0')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')

plt.tight_layout()
plt.show()

## 3
# Edit the script to generate another Sine wave, y1, with frequency F1 and amplitude 1500
# F1 equals to the second partial of the audio wav file <glockenspiel-a-2.wav>. 
# make a new signal by summing y0 and y1 together: y = y0 + y1.
# Generate sine wave y0
F1 = 7012 # To be checked from the previous plot
amplitude = 1500
y1 = amplitude* np.sin(2*np.pi*F1*t_scale)
y=y1+y0
# Plot time domain visualization
t_scale_short = np.linspace(0,len(x)/fs,num=short_length)
amp_scale_short = y[0:short_length]

plt.figure(2)
#plt.subplot(2, 1, 1)
plt.subplot(2, 1, 1)
plt.plot(t_scale_short, amp_scale_short)
plt.title('Time domain visualization of the sine wave (y0+y1)')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time (samples)')

plt.figure(3)
# Plot time domain visualization (on a short segment)
# Plot frequency domain visualization using FFT
Y = scipy.fft(y[t_start:t_start+win_len])
mag_scale = 20*np.log10((Y[0:int(win_len / 2) - 1]))
plt.subplot(2, 1, 2)
plt.plot(frq_scale, mag_scale)
plt.title('Frequency domain visualization of the sine wave y')
plt.axis('tight')
plt.grid('on')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.tight_layout()
plt.show()

## 4
# Record and play sounds. Below is shown how to do for y0, do the same for y1 and y
#coe=np.power(0.01,t_scale)
# To record:
scipy.io.wavfile.write('y0.wav', fs, np.int16(y0))

# To play (on window)
winsound.PlaySound('y0.wav', winsound.SND_FILENAME)
# To play (on linux)
#os.system("aplay y0.wav")
scipy.io.wavfile.write('y.wav', fs, np.int16(y))

# To play (on window)
winsound.PlaySound('y.wav', winsound.SND_FILENAME)