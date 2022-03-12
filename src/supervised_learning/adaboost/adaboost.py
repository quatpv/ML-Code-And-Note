from __future__ import division
from distutils.log import error
import numpy as np
import math

from utils.data_manipulation import train_test_split
from utils.data_operation import accuracy_score


# Decision stump used as weak classifier in this impl of Adaboost
class DecisionStump():
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None
    

class Adaboost():
    """

    Parameters:
    -----------
    n_clf: int
        The number of weak classifiers that will be used.
    """
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        #Initialize weights to 1/N
        w = np.full(n_samples, (1/n_samples))

        self.clfs = []
        # Iterate through classifiers

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')
            # Iterate through every unique feature value and see what value 
            # maskes the best threshold for predicting y
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # Try every unique feature value as threshold
                for threshold in unique_values:
                    p = 1
                    # Initialize all prediction to '1'
                    prediction = np.ones(np.shape(y))
                    # Label the samples whose values are below threshold -1
                    prediction[X[:, feature_i] < threshold] = -1
                    # Error = sum of weights of misclassified samples
                    error = sum(w[y != prediction])

                    # If the error is greater than 50% we flip the polarity so what samples that
                    # were classifier as 0 are classified as 1, and vice versa
                    # E.g error = 0.8 => (1-error) = 0.2

                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    
                    # If this threshold resulted in the smallest error we save the configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error
            
            # Calculate the alpha which is used to update the sample weights,
            # Alpha is also an approximation of this classifier's proficiency
            clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            # Set all predictions to '1' initally
            predictions = np.ones(np.shape(y))
            # The indexes where the sample values are below threshold
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity  * clf.threshold)
            predictions[negative_idx] = -1

            # Calculate new weights
            # Missclassified sample get larger weights and correctly classified samples smaller. 
            w *= np.exp(-clf.alpha * y * predictions)

            # Normalize to one
            w /= np.sum(w)
            



                    






