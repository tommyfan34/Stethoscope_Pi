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

threshold = the threshold to pick up peaks
sp = the threshold of the splitted HS, normally 50 ms
merge = the threshold below which two seperate time gate should be merged into one

Author: Xiao Fan
Date: 7/31/2019
"""

def HSSeg(Pa, wavtime):
    # the threshold to determine the peaks
    threshold = 1
    timegate = np.zeros_like(wavtime)
    # group the time gate into several groups, 'group' is the starting index of each group
    group = []
    for i, yi in enumerate(Pa):
        if yi > threshold:
            timegate[i] = 1
        if i == 0:
            group.append(i)
        elif timegate[i] != timegate[i-1]:
            group.append(i)
    group.append(i+1)
    # find the peaks of each group
    peak = []
    for i, yi in enumerate(group):
        if i != len(group)-1 and timegate[group[i]] == 1:
            locmax = find_locmax(Pa[group[i]:group[i+1]])
            # choose the first locmax as the peak of the group
            peak.append(locmax[0]+group[i])
    return wavtime[peak]

"""
find_locmax() is to help find the maximal value and return the index
"""
def find_locmax(data):
    length = len(data)
    locmax = []
    if length == 1:
        return [0]
    if data[1] < data[0]:
        locmax.append(0)
    for i in range(1, length-1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            locmax.append(i)
    if data[-2] < data[-1]:
        locmax.append(length-1)
    return locmax

if __name__ == "__main__":
    path = ["01 Apex, Normal S1 S2, Supine, Bell_test.wav","a0001.wav","a0002.wav"]
    audio_clip = [0, 5]
    for i, yi in enumerate(path):
        wavdata, wavtime, samplerate = wavread(yi, audio_clip)
        wavdata2, wavtime2 = NASE(wavdata, 0.02 * samplerate, samplerate, audio_clip[0])
        peak = HSSeg(wavdata2, wavtime2)
        figure(i)
        subplot(211)
        xlabel('time(s)')
        ylabel('magnitude')
        title('Before NASE')
        plot(wavtime, wavdata)
        subplot(212)
        xlabel('time(s')
        ylabel('magnitude')
        title("after NASE")
        plot(wavtime2, wavdata2)
        plot(peak, np.ones(len(peak)), 'r+')
    show()