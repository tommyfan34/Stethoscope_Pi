from numpy import *
from numpy.random import *
from scipy import signal
from matplotlib.pyplot import *
from wavread import wavread
from NASE import NASE
import time
"""
This function is to implement adaptive line enhancer with NLMS algorithm
dn = the input signal sequence
M = the number of the response of the FIR filter (rank M-1)
mu = the NLMS step size
un = delayed signal
dn_hat = filtered signal
en = difference between dn and dn_hat
wm = response sequence of the FIR filter, shape = (M,1)
delta = the delayed sample. For the white noise whose all samples are uncorrelated, delta could be 1
xn = the vector to make correlation with e[n], shape = (M,1)
a = alpha coefficient to prevent the denominator of mu become 0

Author: Xiao Fan
Date: 8/1/2019
"""
def ALE_NLMS(dn, M, mu, delta, a):
    # normalize mu
    mu /= M
    # initialize the FIR response sequence
    wm = np.array([zeros(M)]).T
    # initialize error sequence
    en = np.array([zeros_like(dn)]).T
    # initialize output sequence
    dn_hat = np.array([zeros_like(dn)]).T
    # delay the signal by delta
    nominator = [0] * (delta+1)
    nominator[delta] = 1
    un = np.array([signal.lfilter(nominator, 1, dn)]).T  # un is the delayed sequence
    # initialize vector xn for correlation with e[k]
    xn = zeros_like(wm)
    N = len(dn)
    for i, yi in enumerate(np.arange(M, N)):
        # get the xn
        xn = un[yi - 1:yi - M:-1, 0]
        xn = np.append(xn, un[yi - M, 0])
        xn = xn[:, np.newaxis]
        # get the filtered output
        dn_hat[yi - 1, 0] = np.dot(wm.T, xn)
        # form the error sequence
        en[yi - 1, 0] = dn[yi - 1] - dn_hat[yi - 1, 0]
        # update the weight sequence
        wm +=  M * mu / (a + np.dot(xn.T, xn))* en[yi - 1] * xn
    dn_hat = dn_hat.flatten('F')
    return dn_hat, wm, en

if __name__ == "__main__":
    path = ["test3.wav"]
    # SNR is the signal to noise ratio in dB
    SNR = 40
    # crop the .wav file starting from 5 sec to 6 sec
    audio_clip = [1,5]
    for i, yi in enumerate(path):
        start_time = time.time()
        wavdata, wavtime, samplerate = wavread(yi, audio_clip)
        # corrupt the signal with white noise
        noise = sqrt(1/2/(10 ** (SNR/10))) * randn(len(wavdata))
        # noise = zeros_like(wavdata[0])
        # noise[int(samplerate*0.3) : int(samplerate*0.3)+50] = 1
        wavdata_corrupted = wavdata + noise
        dn_hat, wm, en = ALE_NLMS(wavdata,32, 0.016, 1, 0.1)
        wavdata2, wavtime2 = NASE(dn_hat, 0.02 * samplerate, samplerate, audio_clip[0])
        figure(i)
        subplot(311)
        xlabel("time(s)")
        ylabel("normalized magnitude")
        title("original waveform")
        plot(wavtime, wavdata)
        #subplot(412)
        #xlabel("time(s)")
        #ylabel("normalized magnitude")
        #title("corrupted waveform")
        #plot(wavtime, wavdata_corrupted)
        subplot(312)
        xlabel("time(s)")
        ylabel("normalized magnitude")
        title("output waveform")
        plot(wavtime, dn_hat)
        subplot(313)
        xlabel("time(s)")
        ylabel("normalized magnitude")
        title("after NASE")
        plot(wavtime2, wavdata2)
        similarity = np.correlate(wavdata, dn_hat) / np.sqrt(sum(c * c for c in wavdata) * sum(b * b for b in dn_hat))
        print("similarity=", similarity)
        end_time = time.time()
        print("runtime=", end_time-start_time)
    show()




