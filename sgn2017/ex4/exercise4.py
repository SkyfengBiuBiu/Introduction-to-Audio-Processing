
# coding: utf-8

# In[81]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
import scipy
from scipy import signal
from scipy.io import wavfile


## Part 1: Answer the 4 questions as comments in this python file (1 pt)

# The SPL of a signal at frequency band i is 76 dB. The (minimum) masking threshold value for band i is reported from the psychoacoustic model to be 70 dB.


# Q1) How can you evaluate the signal-to-mask ratio (SMR) for this band? What is the value?(0.25pt)
#The SMR at a given frequency is expressed as the difference (in dB) between the SPL of the masker and the masking threshold at that frequency.
#Here, the SMR value should be 76-70=6

# Q2) How many bits you need at least to represent the band’s signal values with, if you can assume that the quantization noise decreases 6 dB by adding one bit? (0.25pt)
#The number of bits that we need to represent is : (76-70)/6=1. So that we need the 1 bit.

# Q3) In the next frame (frame length is here 5 ms) the SPL suddenly drops to 40 dB (almost silent). 
# Based on the knowledge on the masking effect, explain how do you expect the masking threshold value to change from the last frames value (70dB)? 
# Is it closer to 40dB or 70dB? You can assume that adjacent bands are also silent. (0.25pt)
# I aasume that it shall be close to 40 dB.

# Q4) How many uniform quantization steps there are with 6 bits? (0.125pt) Further, if maximum value is one, what is the step-size? (0.125pt)
#The number of uniform quantization steps is : 2^6=64
# If the maximum value is one, the step-size shall be 2*1/64=1/32.


## Part 2: Complete the 2 programming exercises (1 pt)

## Part 2: A) absolute threshold of hearing in quiet environment  (0.5 pt)

# The threshold characterizes the amount of energy needed in a pure tone such that it can be
# detected by a listener in a noiseless environment.
# It is typically expressed in terms of dB SPL. In 1940 Fletcher reported test results for 
# a range of listeners in a national study of typical hearing.
# The quiet threshold for a young listener with acute hearing is approximated by 
# the following non-linear function
# threshold_quiet(f) = 3.64(f/1000)^−0.8 − 6.5*exp(−0.6*(f/1000−3.3)^2) + 10^−3  * (f/1000)^4  (dB SPL)
# (see the reading material in the png file)

# read in the audio file
fs,x = wavfile.read('rhythm_birdland.wav')

# normalize x so that its value is between [-1.00, 1.00] (0.1 pt)
x=x/np.max(np.abs(x))

plt.figure()
plt.plot(x)
plt.show()

winlen_ms=20.0 # milliseconds
winlen=2.0**np.ceil(np.log2(np.float(fs) * winlen_ms/1000.0))

n = int(winlen/2.0+1)
f = np.linspace(1, fs/2, n, endpoint=True)

# calculate SPL in dB versus f in Hz. Implement the threshold_quiet(f) function in line 43. (0.2 pt)
th_quiet = 3.64*(f/1000)**(-0.8) - 6.5*np.exp(-0.6*(f/1000-3.3)**2) + 10**(-3)  * (f/1000)**4

th_quiet[th_quiet>96]=96

# Plot the th_quiet curve. Note the plot is in log scale, help(plt.semilogx) (0.2 pt)
plt.figure()
plt.semilogx(f,th_quiet)
plt.show()


## Part 2: B) SPL Normalization (0.5 pt)

# Procedures:
# Generate a tone at 1kHz for one frame duration, and extract the max level value
# Calculate the maximum absolute value for the DFT of this tone and store the level. (remember to apply windowing!)
# This level is then set represent 96 dB SPL (maximum value for CD SPL), if we start from 0 dB and have 16bits. 
# Normalized_LEVEL = 96 + 20*log10( abs( X_DFT ) / level_at_1khz )
# Analyze the signal with respect to hearing threshold*
# Process a signal using STFT and extract SPL for each time-frequency point: SPL(frame,frequency). 
# See which time-frequency points fall below the hearing threshold. SPL(frame,frequency) < Tq(frequency)
# Plot the time-frequency map of points that fall below the threshold.

# Hann window
h = signal.get_window('hann',int(winlen))

# Generate a tone at 1kHz for one frame duration.
t = np.linspace(0,1,winlen)

# generate the 1kHz sinusoid and extract the max level value
xx = np.sin(1000.0*t*2*pi)
maxlevel1k = np.max(np.abs(np.fft.rfft( h * xx )))
print(maxlevel1k)

# you can use the following template or design your own template 
# to extract SPL and select points below hearing threshold

# Loop through data frame by frame (or use STFT!)
win=-1
si=0
ei=si+int(winlen)

while (ei < len(x)):
    win=win+1
    x_win = x[si:ei] * h
    X = np.fft.rfft(x_win,winlen) 
    # normalized sound pressure level by maxlevel1k. Implement the 'Normalized_LEVEL' equation in line 79 (0.2 pt)
    SPL_frame = 96 + 20*np.log10( np.abs(X) / maxlevel1k )
    # store as matrix
    SPL = SPL_frame if win==0 else np.vstack((SPL_frame,SPL))
    # store DFT values as STFT matrix
    X_dft = X if win==0 else np.vstack((X_dft,X))
    si = si + int(winlen / 2)
    ei = si + int(winlen)
    # store indices (time-frequency) that exceed the threashold of hearing
    over_quit = (SPL_frame > th_quiet).astype('float64')
    SPL_TH = over_quit if win==0 else np.vstack((SPL_TH, over_quit))
       
fv = np.linspace(0,fs/2,n)
tv = np.linspace(0,int(float(len(x))/float(fs)),win+1)

# visualize the SPL (0.1 pt)
plt.figure(figsize=(20, 8))
plt.pcolormesh(tv,fv, np.array(np.asmatrix(SPL).transpose()), cmap='jet')
plt.xlabel('time (frame)')
plt.ylabel('Frequency (index)')
plt.title('log-magnitude spectrogram')
plt.colorbar()
plt.show()
 
# Plot the threshold of hearing exceeding points SPL_TH (0.1 pt)
plt.figure(figsize=(20, 8))
plt.pcolormesh(tv,fv, np.array(np.asmatrix(SPL_TH).transpose()), cmap='jet')
plt.xlabel('time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('log-magnitude spectrogram')
plt.colorbar()
plt.show()

# Remove points in SPL that are below the threshold of hearing SPL_TH (0.1 pt)
plt.figure(figsize=(20, 8))
plt.pcolormesh(tv,fv,np.array(np.asmatrix(SPL*SPL_TH).transpose()), cmap='jet')
plt.xlabel('time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('log-magnitude spectrogram values over the threshold of hearing')
plt.colorbar()
plt.show()


