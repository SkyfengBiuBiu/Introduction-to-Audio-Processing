import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
import scipy
from scipy import signal
from scipy.io import wavfile
from scipy.signal import stft, istft, lfilter
import winsound

#Apply the STFT function on the input data
def STFT(x,fs,winlen,han):
    f, t, X = stft(x, fs=fs, window=han, nperseg=winlen, noverlap=winlen / 2, nfft=winlen, detrend=False,
                   return_onesided=True, padded=True, axis=-1)
    return X

#Apply the inverse STFT function
def iSTFT(X,fs,winlen,han):
    _, x = istft(X, fs=fs, window=han, nperseg=winlen, noverlap=winlen / 2, nfft=winlen, input_onesided=True)
    return x

#Update the harmonic and precussive compoents iteratively and binarize the seperation result
def update(H_updated,P_updated,k,alpha):
    [h_len,i_len]=np.shape(H_updated[0])
    #Execute the ieration
    for k_index in np.arange(k-1):
        H0= H_updated[k_index]
        P0 = P_updated[k_index]
        H1=np.zeros(np.shape(H0))
        P1=np.zeros(np.shape(P0))
        #Calculate the next elemnets with the gradient and previos one
        for h in np.arange(0, h_len-1):
            for i in np.arange(0, i_len-1):
                max=float(np.max([H0[h,i]+gradient(H0,P0,h,i,alpha),0]))
                w=float(W[h,i])
                H1[h,i]=np.minimum(max,w)
                P1[h,i]=W[h,i]-H1[h,i]
        H_updated.append(H1)
        P_updated.append(P1)

    #Take the last elements of the collections
    H_max_1 = H_updated[-1]
    P_max_1 = P_updated[-1]
    H_max = np.zeros(np.shape(H_max_1))
    P_max = np.zeros(np.shape(P_max_1))
    #Excute the binarization process
    for i in np.arange(h_len):
        for j in np.arange(i_len):
            if  H_max_1[i,j]<P_max_1[i,j]:
                H_max[i,j]=0
                P_max[i,j]=W[i,j]
            else:
                H_max[i, j] = W[i, j]
                P_max[i, j] =0
    return [H_max,P_max]

#Calculate the gradient of conrresponding parameters
def gradient(H0,P0,h,i,alpha):
    partH=H0[h,i-1]-2*H0[h,i]+H0[h,i+1]
    partP=P0[h-1,i]-2*P0[h,i]+P0[h+1,i]
    grad=alpha*partH/4-(1-alpha)*partP/4
    return grad

#Plot the images for the results
def plotimages(x,fs,H_max,P_max,h,p,k):
    # Plot the harmonic spectrum
    [n, m] = np.shape(H_max)
    fv = np.linspace(0, fs / 2, n)
    tv = np.linspace(0, int(float(len(x)) / float(fs)), m)
    plt.subplot(2,2,1)
    plt.pcolormesh(tv, fv, H_max, cmap='jet')
    plt.xlabel('time (frame)')
    plt.ylabel('Frequency (index)')
    plt.title('harmonic spectrogram')
    plt.colorbar()

    # Plot the percussive spectrum
    plt.subplot(2, 2, 2)
    plt.pcolormesh(tv, fv, P_max, cmap='jet')
    plt.xlabel('time (frame)')
    plt.ylabel('Frequency (index)')
    plt.title('percussive spectrogram')
    plt.colorbar()

    # Plot time domain visualization of the original signal
    plt.subplot(2, 2, 3)
    t_x = np.linspace(0, int(float(len(x)) / float(fs)), len(x))
    plt.plot(t_x,x)
    plt.xlabel('time')
    plt.ylabel('Frequency')
    plt.title('Original signal ')

    # Plot time domain visualization of the mix signal
    plt.subplot(2, 2, 4)
    m=h+p
    t_m = np.linspace(0, int(float(len(m)) / float(fs)), len(m))
    plt.plot(t_m,m)
    plt.xlabel('time')
    plt.ylabel('Frequency')
    plt.title('Mix signal')
    name = 'seperation_results_k_' + str(k)
    plt.savefig(name)
    plt.show()
    

#Save and play the seperated audio materials
def playSound(fs,h,p,k):
    name = 'harmonic_sqrthan_' + str(k)+'.wav'
    wavfile.write(name, fs, np.asarray(h/np.abs(np.max(h)) * 1000, dtype=np.int16))
    winsound.PlaySound(name, winsound.SND_FILENAME)
    name = 'percussive_sqrthan_' + str(k)+'.wav'
    wavfile.write(name, fs, np.asarray(p/np.abs(np.max(p)) * 1000, dtype=np.int16))
    winsound.PlaySound(name, winsound.SND_FILENAME)

#Calculate the SNR
def evaluate(x,h,p):
    x = x.astype('float64')
    m = (h + p).astype('float64')
    e =x-m[:len(x)]
    x_power=np.sum(np.power(x,2))
    e_power=np.sum(np.power(e,2))
    SNR = 10 * np.log10(x_power / e_power)
    return SNR

#Transform into the waveforms
def convert(H_max, P_max,y):
    H_M = (H_max** (1 / (2 * y))) * np.exp(1j*np.angle(F))
    P_M= (P_max** (1 / (2 * y))) * np.exp(1j*np.angle(F))
    h = iSTFT(H_M, fs, winlen, han)
    p = iSTFT(P_M, fs, winlen, han)
    return [h,p]

if __name__ == "__main__":
    #Load the test material
    fs, x = wavfile.read('rhythm_birdland.wav')
    x=np.asarray(x).astype('float64')

    #Intial the necessary parameters
    y = 0.3
    k=5
    alpha=0.3
    winlen_ms = 20.0  # milliseconds
    winlen = int(2.0 ** np.ceil(np.log2(np.float(fs) * winlen_ms / 1000.0)))
    han = signal.get_window('hann', int(winlen))

    #Call the functions according to the algorithm
    F = STFT(x,fs,winlen,han)
    W = (np.abs(F)) ** (2 * y)
    H = P = W / 2
    H_updated = [H]
    P_updated = [P]
    [H_max, P_max] = update(H_updated,P_updated,k,alpha)
    [h, p] = convert(H_max, P_max,y)
    print("The SNR in this test is "+str(evaluate(x, h, p)))

    #Evaluate through images and seperated audio materials.
    plotimages(x, fs, H_max, P_max, h, p, k)
    playSound(fs, h, p,k)


