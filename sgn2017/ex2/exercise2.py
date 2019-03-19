# coding: utf-8

# import Python modules
from matplotlib.pyplot import *
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from scipy.constants import pi
from scipy.signal import stft, istft, lfilter
from scipy.signal import freqz, convolve
from numpy.fft.fftpack import fft
#import sounddevice as sd


##1. Follow the steps to implement filer banks and answer the questions (0.5 pt).
#!! FILL IN PARTS WITH "None"
 
#(1) load the sample
fs_org,x_org = wavfile.read('gtr55.wav')
x = signal.decimate(x_org,4,ftype='iir')
fs=fs_org/4

#(2) produces an MDCT filter-bank of M channels
def MDCT_FB(M):
    # Length (L) needs to be 2*M (one constraint for PR)
    vn=np.arange(int(M*2))
    print(len(vn))
    # Prototype window (sine window)
    w = np.sin((pi/(4.0*M))*(2.0*vn+1.0)) # window function for perfect reconstruction.
    H_dct=np.zeros((M,2*M))

    for k in range(int(M)): # frequency band index k
        for n in range(int(M*2)): # time sample index n
            H_dct[k,n]=w[n] * np.sqrt(2.0/M) * np.cos( (2.0*n+M+1.0)*(2.0*k+1.0)*pi / (4.0*M) );

    return H_dct

M=25
H = MDCT_FB(M)

#(3) Applies each band filter to the input signal
# empty array for filterbank output
y = np.ndarray([])
# Divide the input signal x into M critically sampled frequency bands.
for k in range(M):
    r =  lfilter(H[k,:],1,x)
    # Downsample (decimate)
    r = r[::M]
    if k==0:
        y = r
    else:
        y = np.vstack((y,r))
    
# - What is the output size of y? (0.1 pt)?
print('{} {}'.format('The output size of y is ',y.size ) )
# - How is the number of bands visible here (0.1 pt)?
#The visible number of bands is 25
# - What is the sampling rate of signals in y in each band (0.1 pt)?
print('{} {}'.format('The sampling rate of signals in y in each band is ',fs/len(y) ) )
        
# - How many samples there are in the original signal "x" and how many samples in "y" (0.1 pt)
print('{} {}'.format('The number of samples in the original signal "x" ', x.size ) )
print('{} {}'.format('The number of samples in the original signal "y" ', y.size ) )
#?
#(4) reconstructs the signal ( combine filterbank output signals into a single signal )
# create a new full sampling rate signal with M channels
x_rec = np.zeros((M,len(x)))
x_rec[:,0::M] = y

for k in range(M):
    # bandpass filter again the subband signals to keep only desired band
    # note TIME-REVERSAL of the filter (flipud)
    x_rec[k,:] =  lfilter(np.flipud(H[k,:]),1,x_rec[k,:])

# WHAT IS THIS SUMMATION DOING? (look at the output dimension of x_rec and compare to x) (0.1 pt)
x_rec = np.sum(x_rec,axis=0)
#The summation is used to sum up of the sub-band frequency.

# Remove the delay the filter caused to the signal (align with input signal)
delay = np.shape(H)[1]-1
x_rec = x_rec[delay::]
x_org = x[:-delay]

# Plot error between reconstruction and original, check that perfect reconstruction happened
plt.figure()
ha,=plt.plot(x_rec[15000:15200],label='reconstructed')
hb,=plt.plot(x[15000:15200],'k--',label='original')
plt.legend(handles=[ha, hb])
plt.show()

##2. Apply an FIR filter to the input signal using convolution(x,filter_coefficients) (0.5 pt).
#!! FILL IN PARTS WITH "None"

# Compute the frequency response of the filter (0.3 pt)
winlen=512
b = np.array([1,-1])
w_,h_=freqz(b,1, int(winlen/2)+1)
plt.figure()
plt.plot(w_, 20*np.log10(np.abs(h_)))
plt.xlabel("Frequency")
plt.ylabel("Amplitude [dB]")
plt.show()

# apply filter to input signal x using convolution in the time-domain (0.2 pt)
x_filter = convolve(x, b)
plt.figure()
plt.plot(x_filter)
plt.show()

##3. STFT and inverse STFT, see the error between reconstruction and original (0.5 pt).
#!! FILL IN PARTS WITH "None"
# Plot the spectrogram using STFT.

win = scipy.signal.hann(winlen, sym=False)
signal.check_COLA(win, winlen, winlen/2)

# Calculate window length (0.1 pt)
print("window length is " + str(float(len(win)/fs*1000)) + ' ms ')

# Take STFT (0.1 pt)
f,t,X = stft(x, fs=fs, window=win, nperseg=winlen, noverlap=winlen/2, nfft=winlen, detrend=False, return_onesided=True, padded=True, axis=-1)

# Apply iSTFT (0.1 pt)
_,x_rec = istft(X,fs=fs,window=win,nperseg=winlen,noverlap=winlen/2,nfft=winlen,input_onesided=True)

# - what is the amount of data, i.e., how many samples are in the STFT domain (0.1 pt)?
print(X.size)
#The number of samples is 20560
# - Each STFT value is complex (one float for real part, one float for imaginary part): 
# - Does the amount of data increase in the STFT domain versus time-domain (0.1 pt)?
##Yes, it does increase as the time increases.


# Calculate reconstruction error
print("reconstruction error is " + str(np.sum(np.abs(x_rec[0:len(x)] - x[0:len(x)]))) )
plt.figure()
ha,=plt.plot(x_rec[15000:15200],label='reconstructed signal part')
hb,=plt.plot(x[15000:15200],'y--',label='original signal')
plt.legend(handles=[ha, hb])
plt.show()

##4. STFT and inverse STFT of the filtered signal (0.5 pt)
#!! FILL IN PARTS WITH "None"
# apply filter h_ to the STFT frames X by multiplication. (0.2 pt)
H = (np.expand_dims(h_,1))
X_H = X*H

# take iSFTF of this filtered signal
_,x_rec_filt = istft(X_H,fs=fs,window=win,nperseg=winlen,noverlap=winlen/2,nfft=winlen,input_onesided=True)

# Look at the reconstruction error
plt.figure()
ha,=plt.plot(x_rec_filt[15000:15200],label='freq-domain filtered signal')
hb,=plt.plot(x_filter[15000:15200],'r--',label='time-domain filtered signal')
plt.legend(handles=[ha, hb])
plt.show()
 
# - what is the amount of data in the original signal (0.1 pt)?
print('{} {}'.format('The number of samples in the original signal ', len(x) ) )

# - What are your observations on the two reconstructed signals in Method 2 and 4 (0.2 pt)?
#ALthough the results are similiar , the STFT could apply the multiplication. In MDCT restruction, the multiplication can not be applied as convolution.