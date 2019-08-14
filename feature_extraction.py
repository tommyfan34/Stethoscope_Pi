import librosa
import librosa.display
from wavread import wavread
from matplotlib.pyplot import *
import pywt
import numpy as np

"""
This function is to extract the time domain & MFCC & DWT feature vector
from a signal

librosa and PyWavelets are pre-installed in conda-channel

Author: Xiao Fan
Date: 8/14/2019
"""
def feature_extraction(sig, samplerate):
    # get the MFCC, window size = 10 ms, stride = 25 ms
    MFCC = librosa.feature.mfcc(y=sig, sr=samplerate, n_mfcc=40, hop_length=int(samplerate*0.01), n_fft=int(samplerate*0.025))
    MFCC = MFCC[1:13, :]
    return MFCC

if __name__ == "__main__":
    path = ["a0004.wav","01 Apex, Normal S1 S2, Supine, Bell_test.wav"]
    audio_clip = [0, 5]
    for i, yi in enumerate(path):
        wavdata, wavtime, samplerate = wavread(yi, audio_clip)
        y, sr = librosa.load(yi, sr=None, duration=5)
        MFCC = feature_extraction(y, sr)
        figure(i, figsize=(10,4))
        librosa.display.specshow(MFCC, x_axis='time', sr=samplerate, hop_length=int(samplerate*0.01))
        colorbar()
        title('MFCC')
        tight_layout()
    show()

