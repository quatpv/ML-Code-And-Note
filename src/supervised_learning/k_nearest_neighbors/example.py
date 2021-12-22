from __future__ import print_function
import numpy as np
from sklearn import datasets

from utils.data_manipulation import train_test_split, normalize
from utils.data_operation import accuracy_score
from src.supervised_learning.k_nearest_neighbors.k_nearest_neighbors import KNN



def main():
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = KNN(k=10)
    y_pred = clf.predict(X_test, X_train, y_train)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    main()
