import numpy as np
from wavread import wavread
from NASE import NASE
from matplotlib.pyplot import *
import math

"""
This is a function to divide the heart sound and identify S1, S2
The input sequence is after normalized average shannon energy (NASE)
The algorithm consists of picking up the peak by determining a threshold, rejecting the extra peaks, identifying the S1 and S2

Input: Pa, the input HS after NASE
wavtime, the time sequence of the input
Output: S1, S2, the sequence of S1 and S2

Author: Xiao Fan
Date: 7/31/2019
"""
def HSSeg(Pa, wavtime):
    # the threshold to determine the peaks
    threshold = 0.8
    timegate = np.zeros_like(wavtime)
    for i, yi in enumerate(Pa):
        if yi > threshold:
            timegate[i] = 1


if __name__ == "__main__":
    path = ["01 Apex, Normal S1 S2, Supine, Bell_test.wav", "02 Apex, Split S1, Supine, Bell.wav"]
    audio_clip = [5, 7]
    for i, yi in enumerate(path):
        wavdata, wavtime, samplerate = wavread(yi, audio_clip)
        wavdata2, wavtime2 = NASE(wavdata[0], 0.02 * samplerate, samplerate, audio_clip[0])
        S1, S2 = HSSeg(wavdata2, wavtime2)
