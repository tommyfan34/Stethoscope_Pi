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
    datause = np.fromstring(datawav, dtype=np.int16)
    # stereo channel
    if channel == 2:
        datause.shape = -1, 2
        datause = datause.T
        time = np.arange(0, nframe) * (1.0/framerate)
        if range != None:
            time = time[int(range[0]*framerate) : int(range[1]*framerate)]
            datause = datause[: , int(range[0]*framerate) : int(range[1]*framerate)]
            # normalize the data
            datause = datause / np.max(np.abs(datause))
        return datause[0], time, framerate
    # mono channel
    elif channel == 1:
        time = np.arange(0, nframe) * (1.0/framerate)
        if range != None:
            time = time[int(range[0] * framerate): int(range[1] * framerate)]
            datause = datause[int(range[0]*framerate) : int(range[1]*framerate)]
            #for i, yi in enumerate(datause):
            #    if abs(yi) >= 6000:
            #        datause[i] = 0
            datause = datause / np.max(np.abs(datause))
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
    path = ["test7.wav"]
    # plot the waveform of the input audio
    for i, n in enumerate(path):
        # read the above
        wavdata, wavtime, samplerate = wavread(n,[1,5])
        plt.figure(i)
        plt.title(n)
        plt.xlabel("time(s)")
        plt.ylabel("magnitude")
        plt.plot(wavtime,wavdata)
    plt.show()


