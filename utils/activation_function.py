import numpy as np

class Sigmoid():
    # https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))
