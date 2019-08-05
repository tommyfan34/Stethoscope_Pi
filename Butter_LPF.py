import numpy as np
from matplotlib.pyplot import *
from scipy import signal
from wavread import wavread
"""
This function is to implement Butterworth low pass filter
"""
def Butter_LPF(un, samplerate, cutoff_freq, order):
    normalized_cutoff = cutoff_freq / (samplerate/2)
    b, a = signal.butter(order, normalized_cutoff, 'low')
    output = signal.filtfilt(b, a, un)
    return output

if __name__ == "__main__":
    path = ["a0001.wav","a0008.wav"]
    audio_clip = [0, 6]
    for i, yi in enumerate(path):
        wavdata, wavtime, samplerate = wavread(yi, audio_clip)
        figure(i)
        subplot(211)
        xlabel('time(s)')
        ylabel('magnitude')
        title('Original waveform')
        plot(wavtime, wavdata)
        wavdata = Butter_LPF(wavdata, samplerate, 500, 5)
        subplot(212)
        xlabel('time(s)')
        ylabel('magnitude')
        title('Filtered waveform')
        plot(wavtime, wavdata)
    show()

