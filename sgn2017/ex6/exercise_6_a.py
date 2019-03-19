# coding: utf-8

# Programming exercise. 1 point in total. Divided between answers and correcting the code.
# NOTE: MATH Question is given in the file: "ex6.pdf"
#(a)0.125
#(b)2.061
#(c) The cosine distance does not change. So I deduce that the ampliﬁcation of signal x(t) or y(t) would not affect this cost.
#(d) After multiplying x1 with 2,the new value is 5.5901699437494745. So the ampliﬁcation of signal x(t) or y(t) would  affect this cost.


import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from dtw import dtw
from numpy.linalg import norm
import librosa
from librosa import display

def new_fig():
    plt.figure(figsize=(20, 3))

def my_draw(x,fs, title_str='Time domain visualization of the audio signal' ):
    new_fig()
    plt.plot(np.linspace(0,len(x)/fs,len(x)),x)
    plt.axis('tight')
    plt.axis((0,float(len(x))/float(fs),np.min(x),np.max(x)))
    plt.grid('on')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title_str)
    plt.show()


import scipy
from scipy.io.wavfile import read

sr1,y1 = read('t2.wav')
sr2,y2 = read('t1_fast.wav')

# integer variables
WINLEN_ms = 50

if sr1 != sr2: # check if sample rates differ
    print("Different " + str(sr1) + str(sr2) )

# processing parameters
WINLEN = int(np.power(2, np.ceil(np.log2(float(WINLEN_ms)/1000.0 * float(sr1)))))
HOPLEN = int(WINLEN/2)

# normalize the input signals using RMS 
y1 = y1/(np.sqrt(np.mean(y1**2)))
y2 = y2/(np.sqrt(np.mean(y2**2)))

# Question 1: What is the duration of signals 1 and 2 in seconds at 0.1s accuracy?
#Answer:
print(len(y1)/sr1)
print(len(y2)/sr2)
## The duration of signals 1 and 2:31.12718820861678 and 28.744965986394558

############# plot time domain signals ##########################

my_draw(y1,sr1, 'signal 1')
my_draw(y2,sr2, 'signal 2')

############### STFT PROCESSING   ##########################

# Let's use Librosa's STFT function.
#
# Question 2: What is the window type applied by default in librosa in the STFT process?
# Hint, See: http://librosa.github.io/librosa/generated/librosa.core.stft.html
#Answer: The default windown is hanning window.
D1 = librosa.stft(y1,n_fft=WINLEN,hop_length=HOPLEN)
D2 = librosa.stft(y2,n_fft=WINLEN,hop_length=HOPLEN)

# Log-mag
log_mag_1 = 20.0 * np.log10( 1e-10 + np.abs ( D1 ))
log_mag_2 = 20.0 * np.log10( 1e-10 + np.abs ( D2 ))

new_fig()
display.specshow(log_mag_1,y_axis='linear', x_axis='time',sr=sr1,hop_length=HOPLEN)
plt.title('Magnitude spectrogram of signal #1')
plt.show()

new_fig()
display.specshow(log_mag_2,y_axis='linear', x_axis='time',sr=sr2, hop_length=HOPLEN)
plt.title('Magnitude spectrogram of signal #2')
plt.show()



########################## MFCCs ##########################

# Let's take MFCC features from the signal 1
#
#
# Question 3: what does the n_mfcc input parameter mean below?
#Answer: The n_mfcc input parameter means number of MFCCs to return.
# (Hint: look at the librosa page http://librosa.github.io/librosa/)
# Use n_mfcc value 40.

# HERE: Take MFCC values of signal 1
mfcc_1 = librosa.feature.mfcc(y1, sr1,n_mfcc=40, n_fft=WINLEN, hop_length=HOPLEN)

# HERE: Take MFCC values of signal 2
mfcc_2 = librosa.feature.mfcc(y2, sr2,n_mfcc=40, n_fft=WINLEN, hop_length=HOPLEN)

# Plot the figures so, that the x-axis is time in seconds, and the y-axis is the MEL frequency.
# See https://librosa.github.io/librosa/generated/librosa.display.specshow.html
# to get the axis labels correct.

new_fig()
display.specshow(mfcc_1, sr=sr1, hop_length=HOPLEN,y_axis='mel', x_axis='time')
plt.title('MFCC features for signal #1')
plt.show()

new_fig()
display.specshow(mfcc_2, sr=sr2, hop_length=HOPLEN,y_axis='mel', x_axis='time')
plt.title('MFCC features for signal #2')
plt.show()

########################## CHROMAGRAMS ##########################

# Take chroma features of signal #1
chroma_1 = librosa.feature.chroma_stft(y=y1, sr=sr1,n_fft=WINLEN,hop_length=HOPLEN )

# HERE: Take chromagram of signal #2
chroma_2 = librosa.feature.chroma_stft(y=y2, sr=sr2,n_fft=WINLEN,hop_length=HOPLEN )

# Plot the chroma vectors.

new_fig()
display.specshow(chroma_1, sr=sr1, hop_length=HOPLEN,y_axis='chroma', x_axis='time')
plt.title('Chroma features for signal #1')
plt.show()

new_fig()
display.specshow(chroma_2, sr=sr2, hop_length=HOPLEN,y_axis='chroma', x_axis='time')
plt.title('Chroma features for signal #2')
plt.show()


# Question 4: What do the the y-axis index represent in the above plots? How many index values are there?
#Answer: The y-axis index represent pitches are determined by the chroma filters.
#There are seven index values there.
##############################################################################

def plot_cost_function_and_optimal_path(f1,f2,len_signal_1,len_signal_2,title_str=None):
    
    f1 = np.where(~np.isnan(f1), f1, 0)
    f2 = np.where(~np.isnan(f2), f2, 0)
    

    # Question 5. What type of distance function is used here in the DTW: Cosine or Euclidean ?
    #Answer: The distance function is Euclidean distance .
    dist, cost, acc_cost, path = dtw(f1.T, f2.T, dist=lambda x, y: norm(x-y,ord=2))
    print ('Normalized distance between the two sounds:' + str(dist))
    plt.figure(figsize=(10.0, 10.0*np.shape(cost)[1]/np.shape(cost)[0] ))
    tv1 = np.linspace(0,len_signal_1,np.shape(f1)[1])
    tv2 = np.linspace(0,len_signal_2,np.shape(f2)[1])
    
    #plt.pcolormesh(tv1,tv2,cost.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest',aspect='auto')
    plt.pcolormesh(tv1,tv2,cost.T)

    plt.plot(path[0]/float(np.shape(f1)[1])*float(len1), path[1]/float(np.shape(f2)[1])*float(len2), 'w')
    #plt.xlim((-0.5, cost.shape[0]-0.5))
    #plt.ylim((-0.5, cost.shape[1]-0.5))
    plt.title(['Cost function and optimal path for features: ' + title_str])
    plt.axis('tight')
    plt.grid('on')
    plt.colorbar()
    plt.xlabel('Signal 1 time (s)')
    plt.ylabel('Signal 2 time (s)')
    plt.show()

len1=float(len(y1))/float(sr1)
len2=float(len(y2))/float(sr2)


# This plots different cost functions with optimal paths.
plot_cost_function_and_optimal_path(mfcc_1,mfcc_2,len1,len2,'MFCC')
plot_cost_function_and_optimal_path(log_mag_1,log_mag_2,len1,len2,'log_mag')
plot_cost_function_and_optimal_path(chroma_1,chroma_2,len1,len2,'chromagram')


# Question 6: What does the color indicate in the matrix plotted?
#The color indicates the cost matrix value of two signals.

# Question 7: Which feature type reveals the repeated structure best? (i.e. which feature has the lowest optimal path loss)
# The  chromagram features reveal the repeated structure best.

# Question 8: For the chromagram feature, besides the optimal path that links synchronizes
# the files, are there any other temporally repeating melodic structure visible in the cost-matrix? How are they visible?
#The parts of picture such as the dark squares repeated melodic structure visible. In this figure, they reveal the matching music notes.





