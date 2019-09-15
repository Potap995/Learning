import numpy as np


def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

def logloss(x, y):
    return - np.dot(y, np.log(x))

def logloss_grad(x, y):
    return -(y / x)

def softmax_grad(x):
    # Let x is col
    x = x.reshape(3, 1)
    return np.diagflat(x) - np.dot(x, x.T)

