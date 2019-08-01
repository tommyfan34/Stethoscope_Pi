import numpy as np
from wavread import wavread
from matplotlib.pyplot import *
import math

"""
NASE(dn) is the function to take normalized average shannon energy on the input sequence
dn is the input raw sequence
N is the signal length in a predetermined time segments
M is the overlapping length
Pa is the output

Date: 7/31/2019   
Author: Xiao Fan
"""
def NASE(dn, N, samplerate,start_time):
    M = N/2
    # normalize the input sequence
    dn = dn / np.max(np.abs(dn))
    for i, yi in enumerate(dn):
        if yi == 0:
            dn[i] = 0
        else:
            dn[i] = - (dn[i] ** 2) * np.log(dn[i] ** 2)
    m = 0
    dn_hat = np.zeros(math.ceil((len(dn)-N)/M))
    for i, yi in enumerate(np.arange(0,len(dn)-N,M)):
        sum = 0
        for k, yk in enumerate(np.arange(yi, yi+N)):
            sum = sum + dn[int(yk)]
        dn_hat[m] = sum / N
        m = m + 1
    Me = np.mean(dn_hat)
    Std = np.std(dn_hat)
    Pa = (dn_hat-Me)/Std
    wavtime = start_time+np.arange(0,m)*M/samplerate
    return Pa, wavtime

if __name__ == "__main__":
    path = ["01 Apex, Normal S1 S2, Supine, Bell_test.wav","02 Apex, Split S1, Supine, Bell.wav"
            ,"03 Apex, S4, LLD, Bell.wav", "04 Apex, Mid Sys Click, Supine, Bell.wav",
            "05 Apex, S3, LLD, Bell.wav",
            "07 Apex, Mid Sys Mur, Supine, Bell.wav",
            "09 Apex, Holo Sys Mur, Supine, Bell.wav","10 Apex, Sys Click & Late Sys Mur, LLD, Bell.wav"]
    # crop the .wav file starting from 5 sec to 6 sec
    audio_clip = [5, 7]
    for i, yi in enumerate(path):
        wavdata, wavtime, samplerate = wavread(yi, audio_clip)
        wavdata2, wavtime2 = NASE(wavdata[0], 0.02*samplerate, samplerate,audio_clip[0])
        figure(i)
        subplot(211)
        xlabel('time(s)')
        ylabel('magnitude')
        title('Before NASE')
        plot(wavtime, wavdata[0])
        subplot(212)
        xlabel('time(s')
        ylabel('magnitude')
        title("after NASE")
        plot(wavtime2, wavdata2)
    show()