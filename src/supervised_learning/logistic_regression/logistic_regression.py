from __future__ import division, print_function
import numpy as np
import math
from utils.data_manipulation import make_diagonal
from utils.activation_functions import Sigmoid
from tqdm import tqdm

class LogisticRegression():
    """ Logistic Regression classifier.
    Parameters:
    learning_rate: float
        The step length that will be taken following the negative gradient during training.
    gradient_descent: boolean
        True or False depending on the gradient descent should be used when training. If False
        then we use batch optimization by least squares.
    """
    def __init__(self, learning_rate=0.1, gradient_descent=True):
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()

    def _init_params(self, X):
        n_features = np.shape(X)[1]
        limit = 1/math.sqrt(n_features)
        # get n_features randomly from uniform distribution [-1/sqrt(N), 1/sqrt(N)]
        self.param = np.random.uniform(-limit, limit, (n_features,))
    
    def fit(self, X, y, n_interations=10000):
        self._init_params(X)
        # Tunning parameters for n_interations
        for i in tqdm(range(n_interations)):
            # Make a new prediction
            y_pred = self.sigmoid(X @ self.param)
            if self.gradient_descent:
                # Using gradient descent to optimize loss function
                self.param -= self.learning_rate * -(y - y_pred) @ X
            else:
                # Make a digonal matrix of the sigmoid gradient cols vector
                diag_gradient = make_diagonal(self.sigmoid.gradient(X @ self.param))
                # Batch option
                self.param = np.linalg.pinv(X.T @ diag_gradient @ X) @ X.T @ (diag_gradient @ X @ self.param + y - y_pred)
    
    def predict(self, X):
        y_pred = np.round(self.sigmoid(X @ self.param)).astype(int)
        return y_pred

    