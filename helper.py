import scipy.io as io
import matplotlib.pyplot as plt
import scipy.signal as sp
import IPython.display as ipd
import numpy as np
import scipy.linalg as lin

def saveSignalAsWAV(name, signal, fs):
    
    io.wavfile.write(name, fs, signal)

def getRecordedSignals():

    path = ["aSignal.wav", "shSignal.wav"]

    fs1, aSignal = io.wavfile.read(path[0])
    fs2, shSignal = io.wavfile.read(path[1])

    aSignal = (aSignal.T)
    shSignal = (shSignal.T)

    fs = {"a": fs1, "sh": fs2}
    signal = {"a": aSignal, "sh": shSignal}

    return fs, signal, path

def play(signal, fs):
    audio = ipd.Audio(signal, rate=fs, autoplay=True)
    return audio

def plot_spectrogram(title, w, fs):
    ff, tt, Sxx = sp.spectrogram(w, fs=fs, nperseg=256, nfft=576)
    fig, ax = plt.subplots()
    ax.pcolormesh(tt, ff, Sxx, cmap='gray_r',
                  shading='gouraud')
    ax.set_title(title)
    ax.set_xlabel('t (sec)')
    ax.set_ylabel('Frequency (Hz)')
    ax.grid(True)

def getNextPowerOfTwo(len):
    return 2**(len*2).bit_length()

def get_optimal_params(x, y, M):
    
    N = len(x)
    r = sp.correlate(x, x)/N
    p = sp.correlate(x, y)/N
    r = r[N-1:N-1 + M]
    p = p[N-1:N-1-(M):-1]           # Correlate calcula la cross-corr r(-k), y necesitamos r(k), y esto no es par como la autocorrelacion
    wo = lin.solve_toeplitz(r, p)

    jo = np.var(y) - np.dot(p, wo)

    NMSE = jo/np.var(y)
    
    return wo, jo, NMSE


