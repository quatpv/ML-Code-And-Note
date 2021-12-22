from __future__ import division, print_function
import numpy as np
from utils.data_operation import euclidean_distance

class KNN():
    """ K nearest neighbors classifier.

    Parameters:
    -----------
    k: int - The number of closest neighbors the will determine the class of sample.
    """
    def __init__(self, k=3):
        self.k = k
    
    def _vote(self, neighbor_labels):
        """ Returns the most common class among the neighbor samples"""
        counts = np.bincount(neighbor_labels.astype(np.int))
        return counts.argmax()
    
    def predict(self, X_test, X_train, y_train):
        y_pred = np.empty(X_test.shape[0])

        for i, test_sample in enumerate(X_test):
            idx = np.argsort([euclidean_distance(test_sample, x) for x in X_train])[:self.k]
            k_nearest_neighbors = np.array([y_train[i] for i in idx])
            y_pred[i] = self._vote(k_nearest_neighbors)

        return y_pred
    