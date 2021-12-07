import numpy as np

def make_diagonal(x):
    """Converts a vector to a diagonal matrix."""
    # https://stackoverflow.com/questions/28598572/how-to-convert-a-column-or-row-matrix-to-a-diagonal-matrix-in-python
    return np.diag(x)