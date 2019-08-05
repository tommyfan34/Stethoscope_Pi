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
    # the low threshold of heart sound noise interval
    HSNoise_LS = 0.15
    # the low threshold of splitted heart sound interval
    HSSplit_LS = 0.05
    # the threshold to determine the peaks
    threshold = 1.1
    # the high threshold interval to recover lost peaks
    HSLost_HS = 0.7
    HSLost_LS = 0.5
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

    # reject the extra peaks
    peak2=peak[:]
    for i, yi in enumerate(wavtime[peak]):
        if (i != len(wavtime[peak])-1) and (timegate[peak[i]]):
            # splitted HS, compare the duration of the two closest time gates
            if interval_timegate(peak[i], peak[i+1], timegate, wavtime) < HSSplit_LS:
                # determine the peak
                if Pa[peak[i+1]]-Pa[find_peak(peak[i], peak2, timegate)] > 0.6:
                    # the former peak is rejected
                    index = peak.index(find_peak(peak[i], peak2, timegate))
                    peak2[index] = -1
                else:
                    peak2[i+1] = -1
                # merge the time gate
                timegate = merge_timegate(peak[i], peak[i + 1], timegate)
            # HS noise, reject the one that has smaller energy, and delete the time gate associated with it
            elif interval_timegate(peak[i], peak[i+1], timegate, wavtime) < HSNoise_LS:
                if Pa[peak[i+1]]-Pa[peak[i]] > 0.6:
                    peak2[i] = -1
                    timegate = delete_timegate(peak[i], timegate)
                else:
                    peak2[i+1] = -1
                    timegate = delete_timegate(peak[i+1], timegate)
    # reject all the extra peaks
    while -1 in peak2:
        peak2.remove(-1)

    # recover the lost peaks
    peak3 = peak2[:]
    for i, yi in enumerate(wavtime[peak2]):
        if i != len(wavtime[peak2])-1:
            #if interval_timegate(peak2[i], peak2[i+1], timegate, wavtime) > HSLost_HS:
            if wavtime[peak2][i+1] - wavtime[peak2][i] > HSLost_HS:
                temp_threshold = threshold
                flag = -1
                # find the start and end index of the interval timegates
                for k, yk in enumerate(timegate[peak2[i]:peak2[i+1] + 1]):
                    if timegate[k + peak2[i]] == 0:
                        if timegate[k - 1 + peak2[i]] == 1:
                            start = k + peak2[i]
                        if timegate[k + 1 + peak2[i]] == 1:
                            end = k + peak2[i]
                # find the highest peak in the interval
                while True:
                    if flag != -1 or temp_threshold < HSLost_LS:
                        break
                    temp_threshold -= 0.01
                    for k in np.arange(start, end+1):
                        if Pa[k] > temp_threshold and wavtime[k] - wavtime[start] > HSNoise_LS and wavtime[end] - wavtime[k] > HSNoise_LS:
                            flag = k
                if flag != -1:
                    timegate[flag] = 1
                    peak3.insert(i+1, flag)

    return wavtime[peak3]

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

"""
The function to merge two timegates given the index of the peaks
"""
def merge_timegate(peak1, peak2, timegate):
    timegate2 = timegate[:]
    for i, yi in enumerate(timegate[peak1:peak2]):
        timegate2[i+peak1] = 1
    return timegate2

"""
The function to determine the interval of two time gates
"""
def interval_timegate(peak1, peak2, timegate,wavtime):
    for i, yi in enumerate(timegate[peak1:peak2+1]):
        if timegate[i+peak1]==0:
            if timegate[i-1+peak1]==1:
                start = i+peak1
            if timegate[i+1+peak1]==1:
                end = i+peak1
    return wavtime[end]-wavtime[start-1]

"""
The function is to find the peak in the time gate
i is the index of the pseudo peak
"""
def find_peak(i, peak, timegate):
    k = i
    while timegate[k] != 0:
        if k in peak:
            return k
        else:
            k -= 1
    k = i
    while timegate[k] != 0:
        if k in peak:
            return k
        else:
            k += 1
    # can't find a peak
    return -1

"""
The function is to delete all the timegate associated with a peak
"""
def delete_timegate(i, timegate):
    k = i
    while timegate[k] == 1:
        timegate[k] = 0
        k -= 1
    k = i
    while timegate[k] == 1:
        timegate[k] = 0
        k += 1
    return timegate

if __name__ == "__main__":
    path = ["a0004.wav","a0002.wav","a0003.wav","a0008.wav","a0005.wav",
            "a0006.wav", "a0007.wav", "a0001.wav"]
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