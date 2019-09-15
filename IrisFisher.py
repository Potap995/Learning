import numpy as np


def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

def logloss(x, y):
    return - np.dot(y, np.log(x).T)

def logloss_grad(x, y):
    return -(y / x)

def softmax_grad(x):
    # Let x is col
    assert x.shape == (1, 3)
    return np.diagflat(x) - np.dot(x.T, x)

