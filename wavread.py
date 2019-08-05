"""
The wavread function is to read .wav file
Mandatory input is the path of the .wav file
Optional input is the desired output time span. Unit: seconds. Its should be in the form of [start, end]
The normalized output datause is in the form of a 2*X darray, while the two rows represent different channels
Author: Xiao Fan @ UCLA
Date: 7/24/2019
"""
import wave
import numpy as np
import matplotlib.pyplot as plt

def wavread(path, range=None):
    # read the file in read-only mode
    wavfile = wave.open(path, "rb")
    # parameters of the wavfile
    wavpara = wavfile.getparams()
    # get the # of channel
    channel = wavfile.getnchannels()
    # get the frame rate and the number of frames
    framerate, nframe = wavpara[2], wavpara[3]
    # read the frame from the data chunk
    datawav = wavfile.readframes(nframe)
    wavfile.close()
    datause = np.fromstring(datawav, dtype=np.short)
    # stereo channel
    if channel == 2:
        datause.shape = -1, 2
        datause = datause.T
        # normalize the data
        datause = datause / np.max(datause)
        time = np.arange(0, nframe) * (1.0/framerate)
        if range != None:
            time = time[int(range[0]*framerate) : int(range[1]*framerate)]
            datause = datause[: , int(range[0]*framerate) : int(range[1]*framerate)]
        return datause[0], time, framerate
    # mono channel
    elif channel == 1:
        datause = datause / np.max(datause)
        time = np.arange(0, nframe) * (1.0/framerate)
        if range != None:
            time = time[int(range[0] * framerate): int(range[1] * framerate)]
            datause = datause[int(range[0]*framerate) : int(range[1]*framerate)]
        return datause, time, framerate

def fft_wav(waveData, plots=True):
    f_array = np.fft.fft(waveData)
    f_abs = f_array
    axis_f = np.linspace(0, 500, np.int(len(f_array)/2))
    if plots == True:
        plt.figure(dpi=100)
        plt.plot(axis_f, np.abs(f_abs[0:len(axis_f)]))
        # plt.plot(axis_f, np.abs(f_abs))
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude spectrum")
        plt.title("Tile map")
        plt.show()
    return f_abs

if __name__ == "__main__":
    path = ["a0001.wav","a0002.wav","01 Apex, Normal S1 S2, Supine, Bell_test.wav"]
    # plot the waveform of the input audio
    for i, n in enumerate(path):
        # read the above
        wavdata, wavtime, samplerate = wavread(n,[5,7])
        fft_size=65536
        fft_size=len(wavdata[0])
        wavdatax = wavdata[0, :fft_size]
        xf=np.fft.fft(wavdatax)/fft_size
        xf=xf[range(int(fft_size/2))]
        xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
        freqz=np.arange(fft_size)*samplerate/fft_size
        freqz=freqz[range(int(fft_size/2))]
        plt.figure(i)
        plt.subplot(211)
        plt.title(n)
        plt.xlabel("time(s)")
        plt.ylabel("Normalized Magnitude")
        plt.plot(wavtime, wavdata)
        plt.subplot(212)
        plt.title("frequency components")
        plt.xlabel("f(Hz)")
        plt.ylabel("Magnitude(dB)")
        plt.plot(freqz,xfp)
    plt.show()


