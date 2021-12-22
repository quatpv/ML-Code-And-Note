from __future__ import division
import numpy as np
import math
import os


def accuracy_score(y_true, y_pred):
    """Compare y_true to y_pred and return the accuracy score"""
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def euclidean_distance(x1, x2):
    """Calculate the l2 distance between two vectors"""
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i])**2
    return np.sqrt(distance)