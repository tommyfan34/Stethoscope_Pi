from sklearn.externals import joblib
from sklearn.svm import SVC
from scipy.stats import skew, kurtosis
from wavread import wavread
from matplotlib.pyplot import *
from HSSeg import HSSeg
from NASE import NASE
from feature_extraction import feature_extraction

"""
This file is to test the SVM model I trained
Date: 8/26/2019
Name: Xiao Fan
"""

if __name__ == "__main__":
    path = ["test.wav"]
    audio_clip = [1, 5]
    for i, yi in enumerate(path):
        wavdata, wavtime, samplerate = wavread(yi, audio_clip)
        wavdata2, wavtime2 = NASE(wavdata, 0.02 * samplerate, samplerate, audio_clip[0])
        peak, s1, s1_start, s1_end, s2, s2_start, s2_end = HSSeg(wavdata2, wavtime2)
        s1_start = ((s1_start - audio_clip[0]) * samplerate).astype(int)
        s1_end = ((s1_end - audio_clip[0]) * samplerate).astype(int)
        s2_start = ((s2_start - audio_clip[0]) * samplerate).astype(int)
        s2_end = ((s2_end - audio_clip[0]) * samplerate).astype(int)
        feature_vector = feature_extraction(wavdata, samplerate, s1_start, s1_end, s2_start, s2_end)
    svm = joblib.load('train_model.pkl')
    result = svm.predict([feature_vector])
    if result == 0:
        print("Your heart sound is normal")
    elif result == 1:
        print("Your heart sound is abnormal")