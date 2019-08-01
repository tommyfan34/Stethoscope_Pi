from numpy import *
from numpy.random import *
from scipy import signal
from matplotlib.pyplot import *
from wavread import wavread
import time

"""
This function is to implement adaptive line enhancer with RLS algorithm
Input:
dn = the input time sequence, shape = (N,)
un = the delayed time sequence, shape = (N,1)
M = the rank of the FIR filter
N = the length of input
P = reverse of auto-correlation matrix, shape = (M,M)
lamda = forget coefficient, constant between 0 and 1, recommend 0.98
delta = constant for initialization, recommends 1e-7
delay = the time delayed
Output: 
dn_hat = the filtered sequence, shape = (N,1)
en = difference between dn and dn_hat, shape = (N,1)
wm = resonse sequence of the FIR filter, shape = (M,1)
xn = the sequence for multiplication, shape = (M,1)
k = gain matrix, shape = (M, 1)

Author: Xiao Fan
Date: 7/26/2019
"""
def ALE_RLS(dn, M, lamda, delta, delay):
    # initialize
    wm = np.array([zeros(M)]).T
    P = np.eye(M) * delta
    N = len(dn)
    #dn_hat=np.array([dn]).T
    dn_hat = np.array([zeros_like(dn)]).T
    en = np.array([zeros_like(dn)]).T
    # delay the signal by "delay"
    nominator = [0] * (delay+1)
    nominator[delay] = 1
    un = np.array([signal.lfilter(nominator, 1, dn)]).T
    xn = zeros_like(wm)

    # RLS adaption
    for i, yi in enumerate(np.arange(M, N)):
        # get the xn
        xn = un[yi-1:yi-M:-1,0]
        xn=np.append(xn,un[yi-M,0])
        xn = xn[:,np.newaxis]
        # get the filtered output
        dn_hat[yi-1,0] = np.dot(wm.T, xn)
        # form the error sequence
        en[yi-1,0] = dn[yi-1] - dn_hat[yi-1,0]
        k = 1/lamda*np.dot(P, xn)/(1+1/lamda*np.dot(xn.T, np.dot(P, xn)))
        P = 1/lamda*P - 1/lamda*np.dot(k, np.dot(xn.T, P))
        wm = wm + (k*np.conj(en[yi-1,0]))
    dn_hat = dn_hat.flatten('F')
    return dn_hat, wm, en

if __name__ == "__main__":
    path = ["a0001.wav","a0002.wav"]
    # SNR is the signal to noise ratio in dB
    SNR = 10
    # crop the .wav file starting from 5 sec to 6 sec
    audio_clip = [0, 5]
    for i, yi in enumerate(path):
        start_time = time.time()
        wavdata, wavtime, samplerate = wavread(yi, audio_clip)
        # corrupt the signal with white noise
        noise = sqrt(1/2/(10 ** (SNR/10))) * randn(len(wavdata))
        # noise = zeros_like(wavdata[0])
        # noise[int(samplerate*0.3) : int(samplerate*0.3)+50] = 1
        wavdata_corrupted = wavdata + noise
        dn_hat, wm, en = ALE_RLS(wavdata_corrupted, 16, 0.95, 1e-3, 2)
        figure(i)
        subplot(311)
        xlabel("time(s)")
        ylabel("normalized magnitude")
        title("original waveform")
        plot(wavtime, wavdata)
        subplot(312)
        xlabel("time(s)")
        ylabel("normalized magnitude")
        title("corrupted waveform")
        plot(wavtime, wavdata_corrupted)
        subplot(313)
        xlabel("time(s)")
        ylabel("normalized magnitude")
        title("output waveform")
        plot(wavtime, dn_hat)
        similarity = np.correlate(wavdata, dn_hat) / np.sqrt(sum(c * c for c in wavdata) * sum(b * b for b in dn_hat))
        print("similarity=", similarity)
        end_time = time.time()
        print("runtime=", end_time-start_time)
    show()


