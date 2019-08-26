from sklearn.externals import joblib
from sklearn.svm import SVC
import csv
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

    X = []
    with open("feature.csv","r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line = list(map(float, line))
            X.append(line)
    svm = joblib.load('train_model.pkl')
    print(svm.predict([X[64]]))