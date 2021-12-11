import numpy as np


def make_diagonal(x):
    """Converts a vector to a diagonal matrix."""
    # https://stackoverflow.com/questions/28598572/how-to-convert-a-column-or-row-matrix-to-a-diagonal-matrix-in-python
    return np.diag(x)


def normalize(X, axis=-1, order=2):
    """Normalize the dataset X"""
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


def shuffle_data(X, y, seed=None):
    """Random shuffle of the samples in X and y"""
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """Split the data to train and test sets"""
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    split_idx = int(len(y) * (1 - test_size))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test