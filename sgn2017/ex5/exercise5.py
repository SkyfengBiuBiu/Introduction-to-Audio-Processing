# coding: utf-8

import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
# to use the librosa package, you need to install it with the command line tool "pip"
# pip install librosa --user 

# Math exercise (1 pt in total)

## Q1. (0.5 pt)
# Consider a vocal tract of length 15cm, and a sound wave traveling through it at 340 m/s.
# How many discrete sampling periods it does take for the whole travel, assuming that the sampling rate is 48 kHz ?

## Q2. (0.5 pt)
# What is the reflection coefficient k when a sound passes from section with area 1cm^2 to 2cm^2?

# Programming exercise (1 pt in total)

## Plot MFCCs

# read in the audio file
fs,x = wavfile.read('gtr55.wav')
#fs,x = wavfile.read('oboe59.wav')
x = signal.decimate(x,4,ftype='iir')
fs = fs/4
total_samples = len(x)
duration = total_samples/float(fs)

# normalize x so that its value is between [-1.00, 1.00]
x = x.astype('float64') / float(numpy.max(numpy.abs(x)))

# MFCCs are useful features in many speech applications.
# use librosa to extract 13 MFCC features
mfccs = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=13)

# Visualize the MFCC series
plt.figure(figsize=(10, 4))
plt.pcolormesh(mfccs)
plt.colorbar()
plt.title('MFCC')
plt.xlabel('Time frame')
plt.ylabel('MFCC index')
plt.tight_layout()
plt.show()

## Q3. (0.5 pt)
# Add the figures with MFCCs for both audio files in your report.

## Pitch tracking

# set a window of duration 20 ms
win_duration = 20
win_length = 2.0**np.ceil(np.log2(np.float(fs) * win_duration/1000.0)) # window length in samples

# set an overlap ratio of 50 %
hop_length = int(win_length/2)

# Compute spectrogram
X = librosa.stft(x, n_fft=int(win_length), hop_length=hop_length)
mag_X = 20 * np.log10(np.abs(X)+0.000001)
number_frequencies, number_time_frames = X.shape
freq_scale = np.linspace(0, fs / 2, number_frequencies)
timeframe_scale = np.linspace(0, duration, number_time_frames)

# extract pitch with librosa
fmin = 80
fmax = 350
threshold_value = 0.13
pitches, magnitudes = librosa.core.piptrack(x, int(fs), n_fft= int(win_length),
                                            hop_length=hop_length, fmin=fmin, fmax=fmax, threshold=threshold_value)

# plot spectrogram
plt.figure(figsize=(20, 22))
plt.pcolormesh(timeframe_scale, freq_scale, mag_X)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

# Post-processing: get the pitches on array form
new_pitches = []
for i in range(0, number_time_frames):
    new_pitches.append(np.max(pitches[:,i]))

# Add the pitch to the spectrogram plot
plt.plot(timeframe_scale, new_pitches, 'm.')
plt.show()

# Get an estimate of the pitch on average over time (use the median function)
pitch_av = np.median(new_pitches)
print(pitch_av)

## Q4. (0.5 pt)
# Add the figures with spectrogram and pitches for both audio files in your report.
# What is the average pitch for both signals?
