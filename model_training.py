from sklearn.externals import joblib
from sklearn import svm
import csv

"""
This function is to train the SVM model
Date: 8/26/2019
Author: Xiao Fan
"""

if __name__ == "__main__":
    X = []
    y = []
    with open("feature.csv","r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line = list(map(float, line))
            X.append(line)
    with open("label.csv","r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            y.extend(line)
        y = list(map(int, y))
    clf = svm.SVC()
    clf.fit(X, y)
    joblib.dump(clf,'train_model.pkl')
