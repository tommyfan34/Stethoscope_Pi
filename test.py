import scipy.io.wavfile
from matplotlib.pyplot import *

def wavread(file):
    samplerate, data = scipy.io.wavfile.read(file)
    return data, samplerate

if __name__ == "__main__":
    path = ["test7.wav"]
    # crop the .wav file starting from 5 sec to 6 sec
    audio_clip = [0, 5]
    for i, yi in enumerate(path):
        data, framerate = wavread(yi)
        figure(i)
        plot(data)
    show()